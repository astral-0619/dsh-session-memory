/**
 * Surface helpers: transcript extraction for the sidechain, tool-call
 * counting, text-item counting, natural-break detection, and node lookup.
 * Port of the counting/estimation helpers in astral-code's tail.rs, adapted
 * to dsh session surfaces (arrays of seqs over an event log).
 * @module dsh-session-memory/transcript
 */

import { createHash } from 'node:crypto'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type { TokenMeasurement, TokenSurfaceNode } from '@deepseek-ai/dsh-token-meter'

/** Events that live on the model-visible surface. */
const SURFACE_EVENT_TYPES = new Set([
  'user/message',
  'assistant/message',
  'tool/result',
  'compaction/summary',
])

export interface SurfaceNodeInfo {
  seq: number
  event: SessionEvent | undefined
  tokens: number
}

/** Map a measurement's nodes back to their events, in surface order. */
export function surfaceNodes(
  session: Session,
  measurement: TokenMeasurement,
): SurfaceNodeInfo[] {
  const bySeq = new Map<number, SessionEvent>()
  for (const event of session.events) bySeq.set(event.seq, event)
  return measurement.nodes.map((node: TokenSurfaceNode) => ({
    seq: node.seq,
    event: bySeq.get(node.seq),
    tokens: node.tokens,
  }))
}

/** Whether the session's last turn ended cleanly (no open tool work). */
export function lastSurfaceEventIsNaturalBreak(session: Session): boolean {
  const surface = session.surface.nodes
  if (surface.length === 0) return true
  const lastSeq = surface[surface.length - 1]!
  const last = session.events.find(event => event.seq === lastSeq)
  if (last === undefined) return true
  return last.type === 'user/message' || last.type === 'assistant/message'
}

/** Count tool calls across the whole surface. */
export function countToolCalls(session: Session): number {
  const surfaceSeqs = new Set(session.surface.nodes)
  let count = 0
  for (const event of session.events) {
    if (event.type === 'tool/call' && surfaceSeqs.has(event.seq)) count += 1
  }
  return count
}

/** Count text-bearing items in a tail seq range (assistant/user messages). */
export function countTextItems(session: Session, seqs: readonly number[]): number {
  let count = 0
  for (const seq of seqs) {
    const event = session.events.find(candidate => candidate.seq === seq)
    if (event === undefined) continue
    if (event.type === 'user/message' || event.type === 'assistant/message') count += 1
  }
  return count
}

function blockText(block: ContentBlock): string {
  switch (block.type) {
    case 'text':
      return block.text
    case 'tool-call':
      return `<tool-call name="${block.name}">${block.arguments}</tool-call>`
    case 'tool-result':
      return `<tool-result>${block.content.map(blockText).join('\n')}</tool-result>`
    default:
      return ''
  }
}

/** Extract a readable transcript line for one surface event. */
export function eventTranscriptLine(event: SessionEvent): string {
  switch (event.type) {
    case 'user/message': {
      const text = event.data.content.map(blockText).join('\n').trim()
      return text.length > 0 ? `User: ${text}` : ''
    }
    case 'assistant/message': {
      const text = event.data.message.content.map(blockText).join('\n').trim()
      return text.length > 0 ? `Assistant: ${text}` : ''
    }
    case 'tool/call':
      return `Tool call: ${event.data.name}(${event.data.arguments})`
    case 'tool/result': {
      const text = event.data.message.content.map(blockText).join('\n').trim()
      return `<tool result> ${text.slice(0, 4000)}`
    }
    case 'compaction/summary': {
      const text = event.data.summary.map(blockText).join('\n').trim()
      return `<previous summary> ${text}`
    }
    default:
      return ''
  }
}

/**
 * Stable fingerprint of a surface node: hash of seq + type + text-ish content.
 * Mirrors astral-code's `item_fingerprint` boundary-stability contract.
 */
export function nodeFingerprint(session: Session, seq: number): string | undefined {
  const event = session.events.find(candidate => candidate.seq === seq)
  if (event === undefined) return undefined
  const content = eventTranscriptLine(event)
  return createHash('sha256')
    .update(`${event.seq}:${event.type}:${content}`)
    .digest('hex')
    .slice(0, 32)
}

/** Token estimate over an arbitrary seq range (sum of meter node prices). */
export function estimateRangeTokens(
  nodes: readonly SurfaceNodeInfo[],
  seqs: readonly number[],
): number {
  const bySeq = new Map<number, number>()
  for (const node of nodes) bySeq.set(node.seq, node.tokens)
  let total = 0
  for (const seq of seqs) total += bySeq.get(seq) ?? 0
  return total
}

export { SURFACE_EVENT_TYPES }
