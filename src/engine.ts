/**
 * SessionMemoryEngine: the dsh `CompactionEngine` implementation backed by
 * the per-session summary file. Port of astral-code's `try_compact` path:
 * summary + verbatim tail replace the shadowed history head.
 * @module dsh-session-memory/engine
 */

import { randomUUID } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import type { CommandId } from '@deepseek-ai/dsh-commands/brand'
import {
  CompactionEngine,
  CompactionId,
  ManualCompactionError,
  compactCheckpointSource,
  toolPairingBalancedAfter,
  toolPairingBalancedBefore,
  type CompactionAgentContext,
  type CompactionResult,
  type CompactionTrigger,
  type ManualCompactAgentContext,
} from '@deepseek-ai/dsh-compaction'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import {
  SessionMemoryStore,
  SummaryMissingError,
  type SessionMemoryState,
} from './store.ts'
import {
  formatSessionMemorySummary,
  selectTail,
  truncateSummaryForCompact,
  validateSummary,
} from './tail.ts'
import {
  countToolCalls,
  estimateRangeTokens,
  surfaceNodes,
  type SurfaceNodeInfo,
} from './transcript.ts'

export interface EngineConfig {
  /** dsh-style pressure threshold: fraction of the routed model's context window. */
  thresholdRatio: number
  /** Summary template (seeds new stores and validates structure). */
  summaryTemplate: string
  /** Root directory holding per-session stores. */
  storeDir: string
  /** Optional path advertised in the compacted summary for the full transcript. */
  transcriptPath?: string
}

/** Port of `inspectCompactionEntryState` (the subset the engine needs). */
interface CompactionEntryState {
  openTurn: number | null
  unmatchedCompactionStart: boolean
}

function inspectEntryState(events: readonly SessionEvent[]): CompactionEntryState {
  let openTurn: number | null = null
  let openTurnKnown = false
  let unmatchedStart = false
  let compactionKnown = false
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (!compactionKnown) {
      if (event.type === 'compaction/start') {
        unmatchedStart = true
        compactionKnown = true
      } else if (event.type === 'compaction/end') {
        compactionKnown = true
      }
    }
    if (!openTurnKnown) {
      if (event.type === 'turn/start') {
        openTurn = event.data.turn
        openTurnKnown = true
      } else if (event.type === 'turn/end') {
        openTurnKnown = true
      }
    }
    if (openTurnKnown && compactionKnown) break
  }
  return { openTurn, unmatchedCompactionStart: unmatchedStart }
}

export class SessionMemoryEngine extends CompactionEngine {
  constructor(
    ctx: Context,
    private readonly config: EngineConfig,
  ) {
    super(ctx)
  }

  private storeFor(session: Session): SessionMemoryStore {
    return new SessionMemoryStore(session.id, `${this.config.storeDir}/${session.id}`)
  }

  /** Resolve the latest routed provider/model, mirroring compaction-basic. */
  private routedTarget(session: Session): { provider: string; model: string } | undefined {
    const header = session.requestHeader()
    if (header === undefined) return undefined
    const config = header.config
    if (config === undefined || config.provider.length === 0 || config.model.length === 0) {
      return undefined
    }
    return { provider: config.provider, model: config.model }
  }

  override async compactIfNeeded(
    agent: CompactionAgentContext,
    trigger: CompactionTrigger,
    signal: AbortSignal,
  ): Promise<CompactionResult | null> {
    const store = this.storeFor(agent.session)

    // Wait for any running sidechain extraction (stale/timeout handling inside).
    let state: SessionMemoryState
    try {
      state = await store.waitForRunningExtraction()
    } catch {
      // Extraction was stale or timed out: record it and refuse to compact
      // this round (the original falls back to the legacy engine, which this
      // preset does not mount — documented deviation).
      return null
    }

    const measurement = this.ctx.tokenMeter.measure(agent.session)
    const nodes = surfaceNodes(agent.session, measurement)

    if (trigger === 'pressure') {
      const target = this.routedTarget(agent.session)
      if (target === undefined) return null
      const modelInfo = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)
      const contextTokens = modelInfo.context?.contextWindow
      if (contextTokens === undefined) return null
      const threshold = Math.floor(contextTokens * this.config.thresholdRatio)
      if (measurement.totalTokens < threshold) return null
    }

    try {
      const summary = await store.readSummary()
      validateSummary(summary, this.config.summaryTemplate)
      const tail = selectTail(agent.session, nodes, state)
      if (tail === null) return null
      if (tail.shadowedSeqs.length === 0) return null
      const first = tail.shadowedSeqs[0]!
      const last = tail.shadowedSeqs[tail.shadowedSeqs.length - 1]!
      return await this.compactRegion(first, last, agent, signal)
    } catch (error) {
      if (error instanceof SummaryMissingError) return null
      this.ctx.logger?.warn(`dsh-session-memory compact refused: ${String(error)}`)
      return null
    }
  }

  override async compactNow(
    agent: ManualCompactAgentContext,
    signal: AbortSignal,
    sourceCommandId?: CommandId,
  ): Promise<CompactionResult | null> {
    signal.throwIfAborted()
    try {
      return await agent.runMaintenance(async (agentSignal) => {
        const operationSignal = AbortSignal.any([agentSignal, signal])
        operationSignal.throwIfAborted()
        const store = this.storeFor(agent.session)
        let state: SessionMemoryState
        try {
          state = await store.waitForRunningExtraction()
        } catch {
          return null
        }
        const measurement = this.ctx.tokenMeter.measure(agent.session)
        const nodes = surfaceNodes(agent.session, measurement)
        const summary = await store.readSummary()
        validateSummary(summary, this.config.summaryTemplate)
        const tail = selectTail(agent.session, nodes, state)
        if (tail === null || tail.shadowedSeqs.length === 0) return null
        const first = tail.shadowedSeqs[0]!
        const last = tail.shadowedSeqs[tail.shadowedSeqs.length - 1]!
        return await this.compactRegionInner(first, last, agent, null, sourceCommandId, operationSignal)
      })
    } catch (error) {
      throw new ManualCompactionError(
        'busy',
        'session-memory manual compaction requires an idle agent',
        { cause: error },
      )
    }
  }

  override async compactRegion(
    start: number,
    end: number,
    agent: CompactionAgentContext,
    signal?: AbortSignal,
  ): Promise<CompactionResult> {
    return this.compactRegionInner(start, end, agent, 'current-turn', undefined, signal)
  }

  /** The transaction body, ported from compaction-basic's `compactSurfaceRegion`. */
  private async compactRegionInner(
    start: number,
    end: number,
    agent: CompactionAgentContext,
    ownerMode: 'current-turn' | null,
    sourceCommandId: CommandId | undefined,
    signal?: AbortSignal,
  ): Promise<CompactionResult> {
    const session = agent.session
    const entry = inspectEntryState(session.events)
    if (entry.unmatchedCompactionStart) {
      throw new Error('session-memory compactRegion: a compaction is already active')
    }
    let owner: number | null
    if (ownerMode === null) {
      if (entry.openTurn !== null) {
        throw new ManualCompactionError(
          'busy',
          'manual compaction: the session already has an open turn',
        )
      }
      owner = null
    } else {
      if (entry.openTurn === null) {
        throw new Error(
          'session-memory compactRegion: no open turn — automatic compaction events must be enclosed in a turn',
        )
      }
      owner = entry.openTurn
    }

    const measurement = this.ctx.tokenMeter.measure(session)
    const nodes = surfaceNodes(session, measurement)
    const surface = session.surface.nodes
    if (!surface.includes(start) || !surface.includes(end)) {
      throw new Error('session-memory compactRegion: range missing from surface')
    }
    const startIdx = surface.indexOf(start)
    const endIdx = surface.indexOf(end)
    if (startIdx < 0 || endIdx < startIdx) {
      throw new Error('session-memory compactRegion: reversed range')
    }
    const shadowedSeqs = surface.slice(startIdx, endIdx + 1)
    if (!toolPairingBalancedBefore(session, start)) {
      throw new Error('session-memory compactRegion: unbalanced left edge')
    }
    if (!toolPairingBalancedAfter(session, end)) {
      throw new Error('session-memory compactRegion: unbalanced right edge')
    }

    const compactionId = CompactionId(randomUUID())
    const lifecycle = {
      compactionId,
      ...(sourceCommandId === undefined ? {} : { sourceCommandId }),
      turn: owner,
    }
    const startEvent = session.append('compaction/start', lifecycle)

    let failure: string | undefined
    try {
      // Build the checkpoint from the persisted summary.
      const store = this.storeFor(session)
      const summary = await store.readSummary()
      validateSummary(summary, this.config.summaryTemplate)
      const { text, wasTruncated } = truncateSummaryForCompact(summary)
      const formatted = formatSessionMemorySummary(
        text,
        wasTruncated,
        this.config.transcriptPath,
        store.summaryPath,
      )
      const checkpointMessage = createUserMessage({
        content: [{ type: 'text', text: formatted }],
        source: compactCheckpointSource(compactionId, sourceCommandId),
      })

      const shadowedTokenCount = estimateRangeTokens(nodes, shadowedSeqs)
      const framedTokens = this.ctx.tokenMeter.estimateMessage(checkpointMessage)
      if (framedTokens >= shadowedTokenCount) {
        throw new Error(
          `summary is not smaller than the shadowed content (${framedTokens} estimated framed tokens >= ${shadowedTokenCount})`,
        )
      }

      // Re-verify the span did not change while we read the summary.
      const after = this.ctx.tokenMeter.measure(session)
      if (JSON.stringify(after.nodes) !== JSON.stringify(measurement.nodes)) {
        throw new Error('session-memory compactRegion: surface changed during summary read')
      }

      const target = this.routedTarget(session)
      const summaryEvent = session.append('compaction/summary', {
        compactionId,
        ...(sourceCommandId === undefined ? {} : { sourceCommandId }),
        summary: [{ type: 'text', text: formatted }],
        shadowedRange: { start, end },
        shadowedSeqs: [...shadowedSeqs],
        shadowedTokenCount,
        provider: target?.provider ?? 'session-memory',
        model: target?.model ?? 'summary.md',
      })

      session.append('user/message', checkpointMessage, {
        surfaceOp: { op: 'replace', start, end },
        sourceEventSeqs: [startEvent.seq, summaryEvent.seq, ...shadowedSeqs],
      })

      const endEvent = session.append('compaction/end', {
        compactionId,
        ...(sourceCommandId === undefined ? {} : { sourceCommandId }),
        turn: owner,
      })

      // Record the post-compact baseline (port of `record_post_compact_baseline`).
      const fresh = this.ctx.tokenMeter.measure(session)
      await store.recordPostCompactBaseline(
        fresh.totalTokens,
        countToolCalls(session),
      )

      return {
        compactionId,
        ...(sourceCommandId === undefined ? {} : { sourceCommandId }),
        startSeq: startEvent.seq,
        summarySeq: summaryEvent.seq,
        endSeq: endEvent.seq,
        summary: [{ type: 'text', text: formatted }],
        shadowedRange: { start, end },
        shadowedSeqs: [...shadowedSeqs],
        shadowedTokenCount,
      }
    } catch (error) {
      failure = error instanceof Error ? error.message : String(error)
      // Best-effort close with the failure recorded.
      session.append('compaction/end', {
        compactionId,
        ...(sourceCommandId === undefined ? {} : { sourceCommandId }),
        turn: owner,
        error: failure,
      })
      throw error
    }
  }
}
