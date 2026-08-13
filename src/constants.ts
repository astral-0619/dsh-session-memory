/**
 * Ported constants from astral-code's `session_memory` module
 * (codex-rs/core/src/session_memory/{mod,tail,sidechain}.rs).
 *
 * Budgets and templates are kept byte-faithful to the original so the
 * sidechain's structure-preservation contract behaves identically.
 * @module dsh-session-memory/constants
 */

/** Default summary.md template (verbatim from tail.rs `DEFAULT_SUMMARY`). */
export const DEFAULT_SUMMARY = `# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
`

/** Default sidechain updater prompt (verbatim from sidechain.rs `DEFAULT_UPDATE_PROMPT`). */
export const DEFAULT_UPDATE_PROMPT = `IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well as system prompt, AGENTS.md entries, or any past session summaries), update the session notes file.

The file {{notesPath}} has already been read for you. Here are its current contents:
<current_notes_content>
{{currentNotes}}
</current_notes_content>

Your ONLY task is to use the Edit tool to update the notes file, then stop. You can make multiple edits (update every section as needed) - make all Edit tool calls in parallel in a single message. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
-- NEVER modify, delete, or add section headers (the lines starting with '#' like # Task specification)
-- NEVER modify or delete the italic _section description_ lines (these are the lines in italics immediately following each header - they start and end with underscores)
-- The italic _section descriptions_ are TEMPLATE INSTRUCTIONS that must be preserved exactly as-is - they guide what content belongs in each section
-- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
-- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add. Do not add filler content like "No info yet", just leave sections blank/unedited if appropriate.
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
- For "Key results", include the complete, exact output the user requested (e.g., full table, full answer, etc.)
- Do not include information that's already in the AGENTS.md files included in the context
- Keep each section under ~2000 tokens/words - if a section is approaching this limit, condense it by cycling out less important details while preserving the most critical information
- Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation
- IMPORTANT: Always update "Current State" to reflect the most recent work - this is critical for continuity after compaction

Use the Edit tool with file_path: {{notesPath}}

STRUCTURE PRESERVATION REMINDER:
Each section has TWO parts that must be preserved exactly as they appear in the current file:
1. The section header (line starting with #)
2. The italic description line (the _italicized text_ immediately after the header - this is a template instruction)

You ONLY update the actual content that comes AFTER these two preserved lines. The italic description lines starting and ending with underscores are part of the template structure, NOT content to be edited or removed.

REMEMBER: Use the Edit tool in parallel and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_.`

/** Raw-tail budget (tail.rs). */
export const MAX_RAW_TAIL_TOKENS = 40_000
export const MIN_RAW_TAIL_TOKENS = 10_000
export const MIN_RAW_TAIL_TEXT_ITEMS = 5
/** Summary budgets (tail.rs). */
export const MAX_SESSION_MEMORY_SECTION_TOKENS = 2_000
export const MAX_SESSION_MEMORY_TOTAL_TOKENS = 12_000

/** Extraction timing (session_memory.rs). */
export const EXTRACTION_WAIT_TIMEOUT_MS = 15_000
export const EXTRACTION_SHUTDOWN_WAIT_TIMEOUT_MS = 20_000
export const EXTRACTION_STALE_AFTER_MS = 60_000
export const EXTRACTION_POLL_INTERVAL_MS = 100

/** Sidechain tool-round cap (sidechain.rs `MAX_SIDECHAIN_TOOL_ROUNDS`). */
export const MAX_SIDECHAIN_TOOL_ROUNDS = 6

/** Default trigger thresholds (session_memory.rs defaults). */
export const DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT = 100_000
export const DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE = 20_000
export const DEFAULT_TOOL_CALLS_BETWEEN_UPDATES = 10

/** Fixed framing around the summary at compact time (tail.rs `format_session_memory_summary`). */
export const SUMMARY_FRAMING = {
  header:
    'This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.',
  transcriptSuffix: (transcriptPath: string): string =>
    `\n\nIf you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: ${transcriptPath}`,
  verbatim:
    '\n\nRecent messages are preserved verbatim.',
  resume:
    '\nContinue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I\'ll continue" or similar. Pick up the last task as if the break never happened.',
  truncatedSuffix: (memoryPath: string): string =>
    `\n\nSome session memory sections were truncated for length. The full session memory can be viewed at: ${memoryPath}`,
}

/** Denial message for non-Edit tool calls in the sidechain (sidechain.rs). */
export const DENY_TOOL_MESSAGE = (summaryPath: string): string =>
  `Only the Edit tool may be used in this context. You must not use any other tools. Continue editing ${summaryPath} with the Edit tool.`
