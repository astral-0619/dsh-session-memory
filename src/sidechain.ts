/**
 * Sidechain extraction: a bounded LLM loop that edits `summary.md` in place.
 * Port of astral-code's sidechain.rs (`run_extraction_inner` +
 * `handle_sidechain_item` + `apply_summary_edit` + `updater_prompt`).
 *
 * Core design, mirroring the original:
 * - The sidechain FORKS the main session's context verbatim: the system
 *   prompt is the agent-scoped assembly rendered exactly as the loop renders
 *   it (`ctx.systemPrompt.assemble(assembleContextFor(agent))` +
 *   `renderPrompt`), and the messages are the session's derived history
 *   (`session.deriveMessages()`) with the updater prompt appended as one
 *   final user message. The prefix is unchanged.
 * - The ONLY delta is the tool surface: `edit` is the sole tool, and any
 *   other tool call is answered with a denial tool result.
 * @module dsh-session-memory/sidechain
 */

import { assembleContextFor, type Agent } from '@deepseek-ai/dsh-agent'
import {
  BlockAssembler,
  CallId,
  createMessage,
  createToolResultMessage,
  createUserMessage,
  type ContentBlock,
  type LlmRuntime,
  type Message,
  type ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { Session } from '@deepseek-ai/dsh-session'
import { renderPrompt, type SystemPrompt } from '@deepseek-ai/dsh-system-prompt'
import {
  DENY_TOOL_MESSAGE,
  MAX_SIDECHAIN_TOOL_ROUNDS,
} from './constants.ts'
import { SessionMemoryStore } from './store.ts'
import { summaryBudgetReminder } from './tail.ts'

export interface ExtractionBoundary {
  seq: number
  fingerprint: string
  tokens: number
  toolCalls: number
}

export interface SidechainOptions {
  agent: Agent
  provider: string
  model: string
  maxRounds: number
  signal?: AbortSignal
}

/**
 * Service references captured eagerly at `apply()` time. The sidechain runs
 * after a turn ends and may outlive the harness context (one-shot drivers
 * dispose the tree right after quiescence); holding the service objects
 * directly keeps the extraction usable without touching the context proxy.
 */
export interface SidechainServices {
  llm: LlmRuntime
  systemPrompt: SystemPrompt
}

const EDIT_TOOL_NAME = 'edit'

/** Port of the sidechain-visible tool surface: Edit only. */
export const EDIT_TOOL_SCHEMA: ToolSchema = {
  name: EDIT_TOOL_NAME,
  description:
    'Edit the session notes file. Replaces one exact old_string occurrence with new_string.',
  parameters: {
    type: 'object',
    properties: {
      file_path: {
        type: 'string',
        description: 'The absolute path of the notes file to edit.',
      },
      old_string: {
        type: 'string',
        description: 'The exact text to replace. Must match exactly once in the file.',
      },
      new_string: {
        type: 'string',
        description: 'The replacement text.',
      },
    },
    required: ['file_path', 'old_string', 'new_string'],
    additionalProperties: false,
  },
}

interface EditArgs {
  file_path?: unknown
  old_string?: unknown
  new_string?: unknown
}

/** Port of `apply_summary_edit`: str_replace against the working text. */
export function applySummaryEdit(
  currentText: string,
  args: EditArgs,
  summaryPath: string,
): { text: string; result: string; edited: boolean } {
  const filePath = typeof args.file_path === 'string' ? args.file_path : ''
  const oldString = typeof args.old_string === 'string' ? args.old_string : ''
  const newString = typeof args.new_string === 'string' ? args.new_string : ''
  if (filePath.trim() === summaryPath.trim()) {
    // Path mismatch tolerated for the sidechain's own file only.
  }
  if (oldString.length === 0) {
    return {
      text: currentText,
      result: 'Error: old_string must be non-empty.',
      edited: false,
    }
  }
  const occurrences = currentText.split(oldString).length - 1
  if (occurrences === 0) {
    return {
      text: currentText,
      result: 'Error: old_string was not found in the file.',
      edited: false,
    }
  }
  if (occurrences > 1) {
    return {
      text: currentText,
      result: `Error: old_string occurs ${occurrences} times — provide more context to make it unique.`,
      edited: false,
    }
  }
  return {
    text: currentText.replace(oldString, newString),
    result: 'Edit applied.',
    edited: true,
  }
}

/** Port of `updater_prompt` + `substitute_prompt_variables`. */
export function buildUpdaterPrompt(
  template: string,
  summaryPath: string,
  currentSummary: string,
  budgetReminder: string,
): string {
  let output = ''
  let rest = template
  for (;;) {
    const start = rest.indexOf('{{')
    if (start < 0) {
      output += rest
      break
    }
    output += rest.slice(0, start)
    const afterOpen = rest.slice(start + 2)
    const end = afterOpen.indexOf('}}')
    if (end < 0) {
      output += rest.slice(start)
      break
    }
    const key = afterOpen.slice(0, end)
    if (key === 'currentNotes') output += currentSummary
    else if (key === 'notesPath') output += summaryPath
    else output += `{{${key}}}`
    rest = afterOpen.slice(end + 2)
  }
  return `${output}${budgetReminder}`
}

/** Port of `run_extraction` + `run_extraction_inner`. */
export async function runExtraction(
  services: SidechainServices,
  session: Session,
  store: SessionMemoryStore,
  updatePromptTemplate: string,
  options: SidechainOptions,
  boundary: ExtractionBoundary,
): Promise<ExtractionBoundary> {
  const currentSummary = await store.readSummary()
  let workingText = currentSummary
  const budgetReminder = summaryBudgetReminder(currentSummary)
  const updaterPrompt = buildUpdaterPrompt(
    updatePromptTemplate,
    store.summaryPath,
    currentSummary,
    budgetReminder,
  )

  // Verbatim context fork: the agent-scoped system assembly rendered exactly
  // as the main loop renders it, plus the session's derived history. The
  // updater prompt is appended as one final user message — the only addition.
  const assembly = await services.systemPrompt.assemble(
    assembleContextFor(options.agent, options.signal),
  )
  const system = renderPrompt(assembly)
  const messages: Message[] = [
    ...session.deriveMessages(),
    createUserMessage({
      content: [{ type: 'text', text: updaterPrompt }],
      source: { kind: 'plugin', plugin: 'dsh-session-memory' },
    }),
  ]

  // Mirror astral's inner loop: process every tool call in a response, then
  // continue to the next round; the extraction ends when a response contains
  // no tool calls (the model's final word). Failed edits surface as tool
  // results so the model can retry, never as an early exit.
  let edited = false
  let round = 0
  for (;;) {
    round += 1
    const assembler = new BlockAssembler()

    const stream = services.llm.stream({
      provider: options.provider,
      model: options.model,
      system,
      messages,
      tools: [EDIT_TOOL_SCHEMA],
      purpose: 'compaction',
      signal: options.signal,
    })
    for await (const chunk of stream) {
      assembler.push(chunk)
    }

    const blocks: ContentBlock[] = assembler.blocks()
    if (process.env.DSH_SESSION_MEMORY_DEBUG !== undefined) {
      console.error(`[dsh-session-memory] sidechain round ${round} blocks: ${JSON.stringify(blocks)} finish: ${JSON.stringify(assembler.finish)}`)
    }
    if (assembler.finish.kind !== 'stop' && assembler.finish.kind !== 'max-tokens') {
      const reason = assembler.finish as { kind: string }
      const detail = 'failure' in reason
        ? JSON.stringify((reason as { failure: unknown }).failure)
        : reason.kind
      throw new Error(`sidechain LLM call ended with ${reason.kind}: ${detail}`)
    }

    // Message-shape contract (mirrors derived history): one assistant message
    // carrying the tool-call blocks, then one tool-result message per call.
    const assistantContent: ContentBlock[] = []
    const results: { callId: string; text: string; isError: boolean }[] = []
    let anyToolCall = false
    for (const block of blocks) {
      if (block.type === 'text' || block.type === 'reasoning') {
        assistantContent.push(block)
        continue
      }
      if (block.type !== 'tool-call') continue
      anyToolCall = true
      assistantContent.push(block)
      if (block.name === EDIT_TOOL_NAME) {
        let args: EditArgs = {}
        try {
          args = JSON.parse(block.arguments) as EditArgs
        } catch {
          args = {}
        }
        const outcome = applySummaryEdit(workingText, args, store.summaryPath)
        workingText = outcome.text
        results.push({ callId: block.id, text: outcome.result, isError: false })
        if (outcome.edited) edited = true
      } else {
        results.push({ callId: block.id, text: DENY_TOOL_MESSAGE(store.summaryPath), isError: true })
      }
    }

    if (!anyToolCall) break
    if (round >= options.maxRounds) {
      throw new Error('session memory extraction exceeded tool-call rounds')
    }
    messages.push(createMessage({
      role: 'assistant',
      content: assistantContent,
      source: {
        kind: 'model',
        provider: options.provider,
        model: options.model,
      },
    }))
    for (const result of results) {
      messages.push(createToolResultMessage({
        callId: CallId(result.callId),
        content: [{ type: 'text', text: result.text }],
        isError: result.isError,
      }))
    }
  }

  if (edited) await store.atomicWriteSummary(workingText)
  await finishExtractionSuccess(store, boundary)
  return boundary
}

async function finishExtractionSuccess(store: SessionMemoryStore, boundary: ExtractionBoundary): Promise<void> {
  await store.finishExtraction({
    seq: boundary.seq,
    fingerprint: boundary.fingerprint,
    tokens: boundary.tokens,
    toolCalls: boundary.toolCalls,
  })
}
