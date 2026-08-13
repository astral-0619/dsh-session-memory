/**
 * Per-session memory store: `summary.md` + `state.json` with atomic writes,
 * exactly mirroring astral-code's `SessionMemoryStore` + `atomic_write`.
 * @module dsh-session-memory/store
 */

import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import {
  EXTRACTION_POLL_INTERVAL_MS,
  EXTRACTION_SHUTDOWN_WAIT_TIMEOUT_MS,
  EXTRACTION_STALE_AFTER_MS,
  EXTRACTION_WAIT_TIMEOUT_MS,
} from './constants.ts'

/** Port of `SessionMemoryState` (serde-default). */
export interface SessionMemoryState {
  last_summary_seq?: number
  last_summary_fingerprint?: string
  last_summary_tokens?: number
  last_summary_tool_calls?: number
  extraction_started_at_unix?: number
  last_error?: string
}

export class ExtractionTimeoutError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ExtractionTimeoutError'
  }
}

export class ExtractionStaleError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ExtractionStaleError'
  }
}

export class SummaryMissingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SummaryMissingError'
  }
}

function nowUnixSeconds(): number {
  return Math.floor(Date.now() / 1000)
}

/** Atomic replace of `path` with `contents` (temp file + rename). */
export async function atomicWrite(path: string, contents: string): Promise<void> {
  const temp = `${path}.tmp-${process.pid}-${cryptoRandomSuffix()}`
  await writeFile(temp, contents, 'utf8')
  try {
    await rename(temp, path)
  } catch (error) {
    await rm(temp, { force: true }).catch(() => {})
    throw error
  }
}

function cryptoRandomSuffix(): string {
  return Math.random().toString(36).slice(2, 10)
}

export class SessionMemoryStore {
  constructor(
    readonly sessionId: string,
    readonly dir: string,
  ) {}

  get summaryPath(): string {
    return join(this.dir, 'summary.md')
  }

  get statePath(): string {
    return join(this.dir, 'state.json')
  }

  /** Create the store if absent; seed the template summary and default state. */
  async ensure(template: string): Promise<void> {
    await mkdir(this.dir, { recursive: true })
    try {
      await readFile(this.summaryPath, 'utf8')
    } catch {
      await atomicWrite(this.summaryPath, template)
    }
    try {
      await readFile(this.statePath, 'utf8')
    } catch {
      await this.writeState({})
    }
  }

  async readSummary(): Promise<string> {
    return readFile(this.summaryPath, 'utf8')
  }

  /** Atomically replace the summary contents (sidechain edit commit). */
  async atomicWriteSummary(contents: string): Promise<void> {
    await atomicWrite(this.summaryPath, contents)
  }

  async readState(): Promise<SessionMemoryState> {
    try {
      const raw = await readFile(this.statePath, 'utf8')
      return JSON.parse(raw) as SessionMemoryState
    } catch {
      return {}
    }
  }

  async writeState(state: SessionMemoryState): Promise<void> {
    await mkdir(dirname(this.statePath), { recursive: true })
    await atomicWrite(this.statePath, `${JSON.stringify(state, null, 2)}\n`)
  }

  /** Mark an extraction as started (state + optional in-memory guard). */
  async markExtractionStarted(): Promise<void> {
    const state = await this.readState()
    state.extraction_started_at_unix = nowUnixSeconds()
    await this.writeState(state)
  }

  /** Record a finished extraction boundary (success) or error (failure). */
  async finishExtraction(
    boundary:
      | {
          seq: number
          fingerprint: string
          tokens: number
          toolCalls: number
        }
      | undefined,
    error?: string,
  ): Promise<void> {
    const state = await this.readState()
    state.extraction_started_at_unix = undefined
    if (boundary !== undefined) {
      state.last_summary_seq = boundary.seq
      state.last_summary_fingerprint = boundary.fingerprint
      state.last_summary_tokens = boundary.tokens
      state.last_summary_tool_calls = boundary.toolCalls
      state.last_error = undefined
    }
    if (error !== undefined) {
      state.last_error = error
    }
    await this.writeState(state)
  }

  /** Clear the boundary tracking after a compact (record new baseline). */
  async recordPostCompactBaseline(tokens: number, toolCalls: number): Promise<void> {
    const state = await this.readState()
    state.last_summary_seq = undefined
    state.last_summary_fingerprint = undefined
    state.last_summary_tokens = tokens
    state.last_summary_tool_calls = toolCalls
    await this.writeState(state)
  }

  /** Poll `state.json` until extraction completes or the timeout expires. */
  async pollForExtractionCompletion(timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs
    for (;;) {
      const state = await this.readState()
      if (state.extraction_started_at_unix === undefined) return true
      if (Date.now() >= deadline) return false
      await new Promise(resolve => setTimeout(resolve, EXTRACTION_POLL_INTERVAL_MS))
    }
  }

  /**
   * Port of `wait_for_running_extraction_with_timeout`: resolve a pending
   * extraction before a compact; stale extractions (60s) are abandoned.
   * Returns the freshest state.
   */
  async waitForRunningExtraction(): Promise<SessionMemoryState> {
    const state = await this.readState()
    const startedAt = state.extraction_started_at_unix
    if (startedAt === undefined) return state
    if (nowUnixSeconds() - startedAt > EXTRACTION_STALE_AFTER_MS / 1000) {
      state.extraction_started_at_unix = undefined
      state.last_error = 'session memory extraction was stale before compact'
      await this.writeState(state)
      throw new ExtractionStaleError(state.last_error)
    }
    const done = await this.pollForExtractionCompletion(EXTRACTION_WAIT_TIMEOUT_MS)
    if (!done) {
      const fresh = await this.readState()
      fresh.extraction_started_at_unix = undefined
      fresh.last_error = 'session memory extraction did not finish before compact timeout'
      await this.writeState(fresh)
      throw new ExtractionTimeoutError(fresh.last_error)
    }
    return this.readState()
  }

  /** Shutdown wait (used by the disposal hook). */
  async waitForExtractionCompletionOnShutdown(): Promise<boolean> {
    return this.pollForExtractionCompletion(EXTRACTION_SHUTDOWN_WAIT_TIMEOUT_MS)
  }
}
