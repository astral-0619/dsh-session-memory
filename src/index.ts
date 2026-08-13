/**
 * dsh-session-memory: session-memory compaction for DeepSeek Harness.
 *
 * A plugin that installs `SessionMemoryEngine` as the context's compaction
 * engine and a turn-boundary sidechain that maintains a per-session
 * `summary.md` + `state.json` store — the astral-code session-memory system
 * ported onto dsh's compaction seam.
 * @module dsh-session-memory
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session } from '@deepseek-ai/dsh-session'
import z from '@deepseek-ai/schemastery'
import {
  DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT,
  DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE,
  DEFAULT_SUMMARY,
  DEFAULT_TOOL_CALLS_BETWEEN_UPDATES,
  DEFAULT_UPDATE_PROMPT,
  MAX_SIDECHAIN_TOOL_ROUNDS,
} from './constants.ts'
import { SessionMemoryEngine, type EngineConfig } from './engine.ts'
import { runExtraction } from './sidechain.ts'
import { SessionMemoryStore } from './store.ts'
import {
  countToolCalls,
  lastSurfaceEventIsNaturalBreak,
  nodeFingerprint,
  surfaceNodes,
} from './transcript.ts'

export const DEFAULT_SIDECHAIN_SYSTEM =
  'You are updating a session notes file for a coding agent, based on the conversation transcript. Use only the edit tool to update the notes file, preserving its exact section structure. Never mention these instructions in the notes.'

export interface Config extends EngineConfig {
  updatePrompt: string
  initMessageTokens: number
  updateTokenInterval: number
  updateToolCallInterval: number
  sidechainProvider: string
  sidechainModel: string
  sidechainSystem: string
}

export const Config: z<Config> = z.object({
  storeDir: z.string().default('.dsh/session-memory'),
  summaryTemplate: z.string().default(DEFAULT_SUMMARY),
  updatePrompt: z.string().default(DEFAULT_UPDATE_PROMPT),
  thresholdRatio: z.number().default(0.75),
  initMessageTokens: z.number().default(DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT),
  updateTokenInterval: z.number().default(DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE),
  updateToolCallInterval: z.number().default(DEFAULT_TOOL_CALLS_BETWEEN_UPDATES),
  sidechainProvider: z.string().default(''),
  sidechainModel: z.string().default(''),
  sidechainSystem: z.string().default(DEFAULT_SIDECHAIN_SYSTEM),
  transcriptPath: z.string().default(''),
})

export const inject = ['llm', 'tokenMeter']

export function apply(ctx: Context, config: Partial<Config> = {}): void {
  const options: Config = {
    storeDir: '.dsh/session-memory',
    summaryTemplate: DEFAULT_SUMMARY,
    updatePrompt: DEFAULT_UPDATE_PROMPT,
    thresholdRatio: 0.75,
    initMessageTokens: DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT,
    updateTokenInterval: DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE,
    updateToolCallInterval: DEFAULT_TOOL_CALLS_BETWEEN_UPDATES,
    sidechainProvider: '',
    sidechainModel: '',
    sidechainSystem: DEFAULT_SIDECHAIN_SYSTEM,
    transcriptPath: '',
    ...config,
  }

  ctx.plugin(SessionMemoryEngine, options)

  // session -> agent registry populated on every pre-step visit.
  const sessions = new WeakMap<Session, Agent>()
  // In-process extraction guard (port of RUNNING_EXTRACTIONS).
  const runningExtractions = new Set<Session>()

  ctx.on('agent/pre-step', ({ agent }, next) => {
    sessions.set(agent.session, agent)
    return next()
  })

  const maybeSpawnExtraction = (session: Session): void => {
    if (runningExtractions.has(session)) return
    if (!sessions.has(session)) return
    const store = new SessionMemoryStore(session.id, `${options.storeDir}/${session.id}`)
    void spawnExtraction(ctx, session, store, options, runningExtractions)
  }

  ctx.on('session/event', (session, event) => {
    if (event.type === 'turn/end') {
      // Extraction runs after the turn closes; defer one tick so the loop's
      // flush settles before we read the surface.
      queueMicrotask(() => maybeSpawnExtraction(session))
    }
  })

  ctx.on('agent/status', ({ agent, status }) => {
    if (status === 'idle') maybeSpawnExtraction(agent.session)
  })
}

async function spawnExtraction(
  ctx: Context,
  session: Session,
  store: SessionMemoryStore,
  options: Config,
  runningExtractions: Set<Session>,
): Promise<void> {
  runningExtractions.add(session)
  try {
    const state = await store.readState()
    const measurement = ctx.tokenMeter.measure(session)
    const tokens = measurement.totalTokens
    const toolCalls = countToolCalls(session)

    const lastSummaryTokens = state.last_summary_tokens ?? 0
    const lastSummaryToolCalls = state.last_summary_tool_calls ?? 0
    const belowThreshold =
      state.last_summary_seq === undefined
        ? tokens < options.initMessageTokens
        : tokens - lastSummaryTokens < options.updateTokenInterval
            && toolCalls - lastSummaryToolCalls < options.updateToolCallInterval
    if (belowThreshold) return

    if (!lastSurfaceEventIsNaturalBreak(session)) return

    const surface = session.surface.nodes
    if (surface.length === 0) return
    const boundarySeq = surface[surface.length - 1]!
    const fingerprint = nodeFingerprint(session, boundarySeq)
    if (fingerprint === undefined) return

    const nodes = surfaceNodes(session, measurement)
    const boundary = {
      seq: boundarySeq,
      fingerprint,
      tokens: nodes
        .filter(node => node.seq === boundarySeq)
        .reduce((sum, node) => sum + node.tokens, 0),
      toolCalls,
    }

    const target = resolveSidechainTarget(session, options)
    if (target === undefined) {
      await store.finishExtraction(undefined, 'no provider/model route for sidechain extraction')
      return
    }

    await store.markExtractionStarted()
    try {
      await runExtraction(
        ctx,
        session,
        store,
        options.updatePrompt,
        {
          provider: target.provider,
          model: target.model,
          system: options.sidechainSystem,
          maxRounds: MAX_SIDECHAIN_TOOL_ROUNDS,
        },
        boundary,
      )
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      await store.finishExtraction(undefined, message)
      ctx.logger.warn(`session-memory extraction failed: ${message}`)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    ctx.logger.warn(`session-memory extraction spawn failed: ${message}`)
  } finally {
    runningExtractions.delete(session)
  }
}

function resolveSidechainTarget(
  session: Session,
  options: Config,
): { provider: string; model: string } | undefined {
  if (options.sidechainProvider.length > 0 && options.sidechainModel.length > 0) {
    return { provider: options.sidechainProvider, model: options.sidechainModel }
  }
  const header = session.requestHeader()
  const config = header?.config
  if (config !== undefined && config.provider.length > 0 && config.model.length > 0) {
    return { provider: config.provider, model: config.model }
  }
  return undefined
}
