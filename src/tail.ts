/**
 * Tail selection, boundary validation, budget checks, and summary
 * truncation — a TypeScript port of astral-code's tail.rs.
 * @module dsh-session-memory/tail
 */

import type { Session } from '@deepseek-ai/dsh-session'
import {
  MAX_RAW_TAIL_TOKENS,
  MAX_SESSION_MEMORY_SECTION_TOKENS,
  MAX_SESSION_MEMORY_TOTAL_TOKENS,
  MIN_RAW_TAIL_TEXT_ITEMS,
  MIN_RAW_TAIL_TOKENS,
  SUMMARY_FRAMING,
} from './constants.ts'
import type { SessionMemoryState } from './store.ts'
import {
  countTextItems,
  estimateRangeTokens,
  nodeFingerprint,
  surfaceNodes,
  type SurfaceNodeInfo,
} from './transcript.ts'

/** Rough token estimate: chars / 4 (matches the dsh heuristic family). */
export function approxTokenCount(text: string): number {
  return Math.ceil(text.length / 4)
}

/** Port of `validate_summary`. */
export function validateSummary(summary: string, template: string): void {
  const trimmed = summary.trim()
  if (trimmed.length === 0 || trimmed === template.trim()) {
    throw new Error('session memory summary is missing or still the template')
  }
}

/** Port of `validate_tail_budget`. */
export function validateTailBudget(tokens: number): void {
  if (tokens > MAX_RAW_TAIL_TOKENS) {
    throw new Error(`session memory raw tail exceeds ${MAX_RAW_TAIL_TOKENS} tokens`)
  }
}

/** Port of `summary_budget_reminder`. */
export function summaryBudgetReminder(summary: string): string {
  const totalTokens = approxTokenCount(summary)
  const oversized = oversizedSections(summary)
  if (totalTokens <= MAX_SESSION_MEMORY_TOTAL_TOKENS && oversized.length === 0) {
    return ''
  }
  let reminder = ''
  const overBudget = totalTokens > MAX_SESSION_MEMORY_TOTAL_TOKENS
  if (overBudget) {
    reminder += `\n\nCRITICAL: The session memory file is currently ~${totalTokens} tokens, which exceeds the maximum of ${MAX_SESSION_MEMORY_TOTAL_TOKENS} tokens. You MUST condense the file to fit within this budget. Aggressively shorten oversized sections by removing less important details, merging related items, and summarizing older entries. Prioritize keeping "Current State" and "Errors & Corrections" accurate and detailed.`
  }
  if (oversized.length > 0) {
    const heading = overBudget
      ? 'Oversized sections to condense'
      : 'IMPORTANT: The following sections exceed the per-section limit and MUST be condensed'
    reminder += `\n\n${heading}:\n${oversized.join('\n')}`
  }
  return reminder
}

/** Port of `summarize_oversized_sections` + `collect_oversized_section`. */
function oversizedSections(summary: string): string[] {
  const sections: string[] = []
  let currentHeader = ''
  let currentLines: string[] = []
  for (const line of summary.split('\n')) {
    if (line.startsWith('#')) {
      collectOversizedSection(sections, currentHeader, currentLines)
      currentHeader = line
      currentLines = []
    } else {
      currentLines.push(line)
    }
  }
  collectOversizedSection(sections, currentHeader, currentLines)
  return sections
}

function collectOversizedSection(
  sections: string[],
  header: string,
  lines: readonly string[],
): void {
  if (header.length === 0) return
  const tokens = approxTokenCount(lines.join('\n'))
  if (tokens > MAX_SESSION_MEMORY_SECTION_TOKENS) {
    sections.push(`- "${header}" is ~${tokens} tokens (limit: ${MAX_SESSION_MEMORY_SECTION_TOKENS})`)
  }
}

/** Port of `truncate_summary_for_compact`: shrink to total budget. */
export function truncateSummaryForCompact(summary: string): { text: string; wasTruncated: boolean } {
  let text = summary
  let tokens = approxTokenCount(text)
  if (tokens <= MAX_SESSION_MEMORY_TOTAL_TOKENS) return { text, wasTruncated: false }
  // Section-aware truncation: cut the largest sections down iteratively.
  for (let pass = 0; pass < 3 && tokens > MAX_SESSION_MEMORY_TOTAL_TOKENS; pass += 1) {
    const sections = splitSections(text)
    if (sections.length === 0) break
    let largest = sections[0]!
    for (const section of sections) {
      if (section.bodyTokens > largest.bodyTokens) largest = section
    }
    if (largest.bodyTokens <= MAX_SESSION_MEMORY_SECTION_TOKENS) break
    const body = truncateBody(largest.body, Math.floor(MAX_SESSION_MEMORY_SECTION_TOKENS * 0.7))
    text = text.replace(largest.body, body)
    tokens = approxTokenCount(text)
  }
  // Absolute fallback: hard cut from the tail end of the file.
  if (tokens > MAX_SESSION_MEMORY_TOTAL_TOKENS) {
    const budget = MAX_SESSION_MEMORY_TOTAL_TOKENS * 4
    text = text.slice(0, budget)
  }
  return { text, wasTruncated: true }
}

interface Section {
  header: string
  body: string
  bodyTokens: number
}

function splitSections(text: string): Section[] {
  const lines = text.split('\n')
  const sections: Section[] = []
  let currentHeader = ''
  let currentBody: string[] = []
  for (const line of lines) {
    if (line.startsWith('#')) {
      if (currentHeader.length > 0) {
        const body = currentBody.join('\n')
        sections.push({ header: currentHeader, body, bodyTokens: approxTokenCount(body) })
      }
      currentHeader = line
      currentBody = []
    } else {
      currentBody.push(line)
    }
  }
  if (currentHeader.length > 0) {
    const body = currentBody.join('\n')
    sections.push({ header: currentHeader, body, bodyTokens: approxTokenCount(body) })
  }
  return sections
}

function truncateBody(body: string, charBudget: number): string {
  if (body.length <= charBudget) return body
  const cut = body.slice(0, charBudget)
  const lastNewline = cut.lastIndexOf('\n')
  return `${cut.slice(0, lastNewline > 0 ? lastNewline : charBudget)}\n_[truncated]_`
}

/** Port of `format_compact_summary` (analysis/summary tag cleanup). */
export function formatCompactSummary(summary: string): string {
  let formatted = summary
  const analysisStart = formatted.indexOf('<analysis>')
  if (analysisStart >= 0) {
    const analysisEnd = formatted.indexOf('</analysis>', analysisStart)
    if (analysisEnd >= 0) {
      formatted = formatted.slice(0, analysisStart) + formatted.slice(analysisEnd + '</analysis>'.length)
    }
  }
  const summaryStart = formatted.indexOf('<summary>')
  if (summaryStart >= 0) {
    const contentStart = summaryStart + '<summary>'.length
    const summaryEnd = formatted.indexOf('</summary>', contentStart)
    if (summaryEnd >= 0) {
      formatted = `${formatted.slice(0, summaryStart)}Summary:\n${formatted.slice(contentStart, summaryEnd).trim()}${formatted.slice(summaryEnd + '</summary>'.length)}`
    }
  }
  while (formatted.includes('\n\n\n')) formatted = formatted.replaceAll('\n\n\n', '\n\n')
  return formatted.trim()
}

/** Port of `format_session_memory_summary`. */
export function formatSessionMemorySummary(
  summary: string,
  wasTruncated: boolean,
  transcriptPath: string | undefined,
  memoryPath: string,
): string {
  const formattedSummary = formatCompactSummary(summary)
  let formatted = `${SUMMARY_FRAMING.header}\n\n${formattedSummary}`
  if (transcriptPath !== undefined) {
    formatted += SUMMARY_FRAMING.transcriptSuffix(transcriptPath)
  }
  formatted += SUMMARY_FRAMING.verbatim
  formatted += SUMMARY_FRAMING.resume
  if (wasTruncated) {
    formatted += SUMMARY_FRAMING.truncatedSuffix(memoryPath)
  }
  return formatted
}

/**
 * Port of `raw_tail_after_summary_boundary` + `calculate_tail_start` +
 * `adjust_start_to_preserve_pairs`.
 *
 * Returns the tail seqs retained verbatim after a compact, or `null` when the
 * boundary cannot be located (fingerprint mismatch or missing summary state).
 */
export function selectTail(
  session: Session,
  nodes: readonly SurfaceNodeInfo[],
  state: SessionMemoryState,
): { tailSeqs: number[]; shadowedSeqs: number[]; tokens: number } | null {
  const surface = session.surface.nodes
  const surfaceSet = new Set(surface)

  // Floor: after the last compaction boundary marker on the surface.
  let floorIndex = 0
  for (let index = surface.length - 1; index >= 0; index -= 1) {
    const event = nodes.find(node => node.seq === surface[index])?.event
    if (event !== undefined && event.type === 'compaction/summary') {
      floorIndex = index + 1
      break
    }
  }

  // Boundary: last_summary_seq, fingerprint-verified.
  let startIndex: number
  if (state.last_summary_seq === undefined) {
    startIndex = surface.length
  } else {
    const boundaryIndex = surface.indexOf(state.last_summary_seq)
    if (boundaryIndex < 0) {
      throw new Error('session memory boundary not found')
    }
    const fingerprint = state.last_summary_fingerprint
    if (fingerprint === undefined) {
      throw new Error('session memory boundary fingerprint missing')
    }
    const actual = nodeFingerprint(session, state.last_summary_seq)
    if (actual !== fingerprint) {
      throw new Error('session memory boundary fingerprint mismatch')
    }
    startIndex = boundaryIndex + 1
  }

  // Grow the tail backward while it is under the minimum size, capped by max.
  let start = Math.max(startIndex, floorIndex)
  let tail = surface.slice(start)
  let tokens = estimateRangeTokens(nodes, tail)
  let textItems = countTextItems(session, tail)
  while (start > floorIndex && (tokens < MIN_RAW_TAIL_TOKENS || textItems < MIN_RAW_TAIL_TEXT_ITEMS)) {
    const candidateSeq = surface[start - 1]!
    const candidateTokens = estimateRangeTokens(nodes, [candidateSeq])
    if (tokens > 0 && tokens + candidateTokens > MAX_RAW_TAIL_TOKENS) break
    start -= 1
    tail = surface.slice(start)
    tokens += candidateTokens
    textItems += countTextItems(session, [candidateSeq])
  }

  // Pull tool calls whose results live inside the tail back into it.
  const resultCallIds = new Set<string>()
  for (const seq of tail) {
    const event = nodes.find(node => node.seq === seq)?.event
    if (event !== undefined && event.type === 'tool/result') {
      resultCallIds.add(event.data.message.content[0]?.toolCallId ?? '')
    }
  }
  if (resultCallIds.size > 0) {
    for (let index = start - 1; index >= floorIndex; index -= 1) {
      const event = nodes.find(node => node.seq === surface[index])?.event
      if (event !== undefined && event.type === 'tool/call' && resultCallIds.has(event.data.callId)) {
        start = index
      }
    }
  }

  tail = surface.slice(start)
  tokens = estimateRangeTokens(nodes, tail)
  validateTailBudget(tokens)

  const shadowedSeqs = surface.slice(0, start).filter(seq => surfaceSet.has(seq))
  if (shadowedSeqs.length === 0) return null
  return { tailSeqs: tail, shadowedSeqs, tokens }
}
