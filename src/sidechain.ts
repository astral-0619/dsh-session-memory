/**
 * Sidechain extraction: a bounded LLM loop that edits `summary.md` in place.
 * Port of astral-code's sidechain.rs (`run_extraction_inner` +
 * `handle_sidechain_item` + `apply_summary_edit` + `updater_prompt`).
 *
 * Deviations from the original (reported):
 * - The original replays the MAIN session prompt template as the sidechain's
 *   system prompt; dsh exposes no cheap way to read the assembled system
 *   prompt, so the sidechain uses a small fixed system prompt instead
 *   (configurable via `sidechainSystem`).
 * - The original executes the Codex Edit tool against the file; this port
 *   implements an equivalent str_replace executor in-process.
 * @module dsh-session-memory/sidechain
 */

import type { Context } from '@deepseek-ai/cordis'
import {
  BlockAssembler,
  createMessage,
  createUserMessage,
  type ContentBlock,
  type Message,
  type ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { Session } from '@deepseek-ai/dsh-session'
import {
  DENY_TOOL_MESSAGE,
  MAX_SIDECHAIN_TOOL_ROUNDS,
} from './constants.ts'
import { SessionMemoryStore } from './store.ts'
import { summaryBudgetReminder } from './tail.ts'
import { buildTranscript } from './transcript.ts'

export interface ExtractionBoundary {
  seq: number
  fingerprint: string
  tokens: number
  toolCalls: number
}

export interface SidechainOptions {
  provider: string
  model: string
  system?: string
  maxRounds: number
  signal?: AbortSignal
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
  ctx: Context,
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
  const transcript = buildTranscript(session)

  const messages: Message[] = [
    createUserMessage({
      content: [{ type: 'text', text: `<conversation>\n${transcript}\n</conversation>` }],
      source: { kind: 'plugin', plugin: 'dsh-session-memory' },
    }),
    createUserMessage({
      content: [{ type: 'text', text: updaterPrompt }],
      source: { kind: 'plugin', plugin: 'dsh-session-memory' },
    }),
  ]

  for (let round = 0; round < options.maxRounds; round += 1) {
    let needsFollowUp = false
    let editedSummary = false
    let anyToolCall = false
    const assembler = new BlockAssembler()

    const stream = ctx.llm.stream({
      provider: options.provider,
      model: options.model,
      system: options.system,
      messages,
      tools: [EDIT_TOOL_SCHEMA],
      purpose: 'compaction',
      signal: options.signal,
    })
    for await (const chunk of stream) {
      assembler.push(chunk)
    }

    const blocks: ContentBlock[] = assembler.blocks()
    const assistantBlocks: ContentBlock[] = []
    for (const block of blocks) {
      if (block.type === 'tool-call') {
        anyToolCall = true
        assistantBlocks.push(block)
        if (block.name === EDIT_TOOL_NAME) {
          let args: EditArgs = {}
          try {
            args = JSON.parse(block.arguments) as EditArgs
          } catch {
            args = {}
          }
          const outcome = applySummaryEdit(workingText, args, store.summaryPath)
          workingText = outcome.text
          assistantBlocks.push({
            type: 'tool-result',
            toolCallId: block.id,
            content: [{ type: 'text', text: outcome.result }],
          })
          if (outcome.edited) editedSummary = true
        } else {
          needsFollowUp = true
          assistantBlocks.push({
            type: 'tool-result',
            toolCallId: block.id,
            content: [{ type: 'text', text: DENY_TOOL_MESSAGE(store.summaryPath) }],
            isError: true,
          })
        }
      } else if (block.type === 'text' || block.type === 'reasoning') {
        // Text content stays local to this extraction; nothing enters notes.
      }
    }

    if (editedSummary) {
      await store.atomicWriteSummary(workingText)
      await finishExtractionSuccess(store, boundary)
      return boundary
    }

    if (anyToolCall) {
      messages.push(createAssistantMessageWithBlocks(assistantBlocks, options))
    }
    if (!needsFollowUp) {
      // Model produced text (or nothing) without further tool work: extraction
      // completes; summary unchanged unless an edit already landed above.
      await finishExtractionSuccess(store, boundary)
      return boundary
    }
  }

  throw new Error('session memory extraction exceeded tool-call rounds')
}

async function finishExtractionSuccess(store: SessionMemoryStore, boundary: ExtractionBoundary): Promise<void> {
  await store.finishExtraction({
    seq: boundary.seq,
    fingerprint: boundary.fingerprint,
    tokens: boundary.tokens,
    toolCalls: boundary.toolCalls,
  })
}

function createAssistantMessageWithBlocks(
  blocks: ContentBlock[],
  options: SidechainOptions,
): Message {
  return createMessage({
    role: 'assistant',
    content: blocks,
    source: {
      kind: 'model',
      provider: options.provider,
      model: options.model,
    },
  })
}
