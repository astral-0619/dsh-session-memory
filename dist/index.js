import z from "@deepseek-ai/schemastery";
import { createHash, randomUUID } from "node:crypto";
import { assembleContextFor } from "@deepseek-ai/dsh-agent";
import { CompactionEngine, CompactionId, ManualCompactionError, compactCheckpointSource, toolPairingBalancedAfter, toolPairingBalancedBefore } from "@deepseek-ai/dsh-compaction";
import { BlockAssembler, CONTEXT_WINDOW_EXCEEDED_CODE, createMessage, createUserMessage } from "@deepseek-ai/dsh-llm";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { renderPrompt } from "@deepseek-ai/dsh-system-prompt";
//#region src/constants.ts
/**
* Ported constants from astral-code's `session_memory` module
* (codex-rs/core/src/session_memory/{mod,tail,sidechain}.rs).
*
* Budgets and templates are kept byte-faithful to the original so the
* sidechain's structure-preservation contract behaves identically.
* @module dsh-session-memory/constants
*/
/** Default summary.md template (verbatim from tail.rs `DEFAULT_SUMMARY`). */
const DEFAULT_SUMMARY = `# Session Title
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
`;
/** Default sidechain updater prompt (verbatim from sidechain.rs `DEFAULT_UPDATE_PROMPT`). */
const DEFAULT_UPDATE_PROMPT = `IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

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

REMEMBER: Use the Edit tool in parallel and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_.`;
/** Raw-tail budget (tail.rs). */
const MAX_RAW_TAIL_TOKENS = 4e4;
/** Summary budgets (tail.rs). */
const MAX_SESSION_MEMORY_SECTION_TOKENS = 2e3;
const MAX_SESSION_MEMORY_TOTAL_TOKENS = 12e3;
const EXTRACTION_SHUTDOWN_WAIT_TIMEOUT_MS = 2e4;
/** Default trigger thresholds (session_memory.rs defaults). */
const DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT = 1e5;
const DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE = 2e4;
/** Fixed framing around the summary at compact time (tail.rs `format_session_memory_summary`). */
const SUMMARY_FRAMING = {
	header: "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.",
	transcriptSuffix: (transcriptPath) => `\n\nIf you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: ${transcriptPath}`,
	verbatim: "\n\nRecent messages are preserved verbatim.",
	resume: "\nContinue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with \"I'll continue\" or similar. Pick up the last task as if the break never happened.",
	truncatedSuffix: (memoryPath) => `\n\nSome session memory sections were truncated for length. The full session memory can be viewed at: ${memoryPath}`
};
/** Denial message for non-Edit tool calls in the sidechain (sidechain.rs). */
const DENY_TOOL_MESSAGE = (summaryPath) => `Only the Edit tool may be used in this context. You must not use any other tools. Continue editing ${summaryPath} with the Edit tool.`;
//#endregion
//#region src/store.ts
/**
* Per-session memory store: `summary.md` + `state.json` with atomic writes,
* exactly mirroring astral-code's `SessionMemoryStore` + `atomic_write`.
* @module dsh-session-memory/store
*/
var ExtractionTimeoutError = class extends Error {
	constructor(message) {
		super(message);
		this.name = "ExtractionTimeoutError";
	}
};
var ExtractionStaleError = class extends Error {
	constructor(message) {
		super(message);
		this.name = "ExtractionStaleError";
	}
};
var SummaryMissingError = class extends Error {
	constructor(message) {
		super(message);
		this.name = "SummaryMissingError";
	}
};
function nowUnixSeconds() {
	return Math.floor(Date.now() / 1e3);
}
/** Atomic replace of `path` with `contents` (temp file + rename). */
async function atomicWrite(path, contents) {
	const temp = `${path}.tmp-${process.pid}-${cryptoRandomSuffix()}`;
	await writeFile(temp, contents, "utf8");
	try {
		await rename(temp, path);
	} catch (error) {
		await rm(temp, { force: true }).catch(() => {});
		throw error;
	}
}
function cryptoRandomSuffix() {
	return Math.random().toString(36).slice(2, 10);
}
var SessionMemoryStore = class {
	sessionId;
	dir;
	constructor(sessionId, dir) {
		this.sessionId = sessionId;
		this.dir = dir;
	}
	get summaryPath() {
		return join(this.dir, "summary.md");
	}
	get statePath() {
		return join(this.dir, "state.json");
	}
	/** Create the store if absent; seed the template summary and default state. */
	async ensure(template) {
		await mkdir(this.dir, { recursive: true });
		try {
			await readFile(this.summaryPath, "utf8");
		} catch {
			await atomicWrite(this.summaryPath, template);
		}
		try {
			await readFile(this.statePath, "utf8");
		} catch {
			await this.writeState({});
		}
	}
	async readSummary() {
		return readFile(this.summaryPath, "utf8");
	}
	/** Atomically replace the summary contents (sidechain edit commit). */
	async atomicWriteSummary(contents) {
		await atomicWrite(this.summaryPath, contents);
	}
	async readState() {
		try {
			const raw = await readFile(this.statePath, "utf8");
			return JSON.parse(raw);
		} catch {
			return {};
		}
	}
	async writeState(state) {
		await mkdir(dirname(this.statePath), { recursive: true });
		await atomicWrite(this.statePath, `${JSON.stringify(state, null, 2)}\n`);
	}
	/** Mark an extraction as started (state + optional in-memory guard). */
	async markExtractionStarted() {
		const state = await this.readState();
		state.extraction_started_at_unix = nowUnixSeconds();
		await this.writeState(state);
	}
	/** Record a finished extraction boundary (success) or error (failure). */
	async finishExtraction(boundary, error) {
		const state = await this.readState();
		state.extraction_started_at_unix = void 0;
		if (boundary !== void 0) {
			state.last_summary_seq = boundary.seq;
			state.last_summary_fingerprint = boundary.fingerprint;
			state.last_summary_tokens = boundary.tokens;
			state.last_summary_tool_calls = boundary.toolCalls;
			state.last_error = void 0;
		}
		if (error !== void 0) state.last_error = error;
		await this.writeState(state);
	}
	/** Clear the boundary tracking after a compact (record new baseline). */
	async recordPostCompactBaseline(tokens, toolCalls) {
		const state = await this.readState();
		state.last_summary_seq = void 0;
		state.last_summary_fingerprint = void 0;
		state.last_summary_tokens = tokens;
		state.last_summary_tool_calls = toolCalls;
		await this.writeState(state);
	}
	/** Poll `state.json` until extraction completes or the timeout expires. */
	async pollForExtractionCompletion(timeoutMs) {
		const deadline = Date.now() + timeoutMs;
		for (;;) {
			if ((await this.readState()).extraction_started_at_unix === void 0) return true;
			if (Date.now() >= deadline) return false;
			await new Promise((resolve) => setTimeout(resolve, 100));
		}
	}
	/**
	* Port of `wait_for_running_extraction_with_timeout`: resolve a pending
	* extraction before a compact; stale extractions (60s) are abandoned.
	* Returns the freshest state.
	*/
	async waitForRunningExtraction() {
		const state = await this.readState();
		const startedAt = state.extraction_started_at_unix;
		if (startedAt === void 0) return state;
		if (nowUnixSeconds() - startedAt > 60) {
			state.extraction_started_at_unix = void 0;
			state.last_error = "session memory extraction was stale before compact";
			await this.writeState(state);
			throw new ExtractionStaleError(state.last_error);
		}
		if (!await this.pollForExtractionCompletion(15e3)) {
			const fresh = await this.readState();
			fresh.extraction_started_at_unix = void 0;
			fresh.last_error = "session memory extraction did not finish before compact timeout";
			await this.writeState(fresh);
			throw new ExtractionTimeoutError(fresh.last_error);
		}
		return this.readState();
	}
	/** Shutdown wait (used by the disposal hook). */
	async waitForExtractionCompletionOnShutdown() {
		return this.pollForExtractionCompletion(EXTRACTION_SHUTDOWN_WAIT_TIMEOUT_MS);
	}
};
//#endregion
//#region src/transcript.ts
/**
* Surface helpers: transcript extraction for the sidechain, tool-call
* counting, text-item counting, natural-break detection, and node lookup.
* Port of the counting/estimation helpers in astral-code's tail.rs, adapted
* to dsh session surfaces (arrays of seqs over an event log).
* @module dsh-session-memory/transcript
*/
/** Map a measurement's nodes back to their events, in surface order. */
function surfaceNodes(session, measurement) {
	const bySeq = /* @__PURE__ */ new Map();
	for (const event of session.events) bySeq.set(event.seq, event);
	return measurement.nodes.map((node) => ({
		seq: node.seq,
		event: bySeq.get(node.seq),
		tokens: node.tokens
	}));
}
/** Whether the session's last turn ended cleanly (no open tool work). */
function lastSurfaceEventIsNaturalBreak(session) {
	const surface = session.surface.nodes;
	if (surface.length === 0) return true;
	const lastSeq = surface[surface.length - 1];
	const last = session.events.find((event) => event.seq === lastSeq);
	if (last === void 0) return true;
	return last.type === "user/message" || last.type === "assistant/message";
}
/** Count tool calls across the whole surface. */
function countToolCalls(session) {
	const surfaceSeqs = new Set(session.surface.nodes);
	let count = 0;
	for (const event of session.events) if (event.type === "tool/call" && surfaceSeqs.has(event.seq)) count += 1;
	return count;
}
/** Count text-bearing items in a tail seq range (assistant/user messages). */
function countTextItems(session, seqs) {
	let count = 0;
	for (const seq of seqs) {
		const event = session.events.find((candidate) => candidate.seq === seq);
		if (event === void 0) continue;
		if (event.type === "user/message" || event.type === "assistant/message") count += 1;
	}
	return count;
}
function blockText(block) {
	switch (block.type) {
		case "text": return block.text;
		case "tool-call": return `<tool-call name="${block.name}">${block.arguments}</tool-call>`;
		case "tool-result": return `<tool-result>${block.content.map(blockText).join("\n")}</tool-result>`;
		default: return "";
	}
}
/** Extract a readable transcript line for one surface event. */
function eventTranscriptLine(event) {
	switch (event.type) {
		case "user/message": {
			const text = event.data.content.map(blockText).join("\n").trim();
			return text.length > 0 ? `User: ${text}` : "";
		}
		case "assistant/message": {
			const text = event.data.message.content.map(blockText).join("\n").trim();
			return text.length > 0 ? `Assistant: ${text}` : "";
		}
		case "tool/call": return `Tool call: ${event.data.name}(${event.data.arguments})`;
		case "tool/result": return `<tool result> ${event.data.message.content.map(blockText).join("\n").trim().slice(0, 4e3)}`;
		case "compaction/summary": return `<previous summary> ${event.data.summary.map(blockText).join("\n").trim()}`;
		default: return "";
	}
}
/**
* Stable fingerprint of a surface node: hash of seq + type + text-ish content.
* Mirrors astral-code's `item_fingerprint` boundary-stability contract.
*/
function nodeFingerprint(session, seq) {
	const event = session.events.find((candidate) => candidate.seq === seq);
	if (event === void 0) return void 0;
	const content = eventTranscriptLine(event);
	return createHash("sha256").update(`${event.seq}:${event.type}:${content}`).digest("hex").slice(0, 32);
}
/** Token estimate over an arbitrary seq range (sum of meter node prices). */
function estimateRangeTokens(nodes, seqs) {
	const bySeq = /* @__PURE__ */ new Map();
	for (const node of nodes) bySeq.set(node.seq, node.tokens);
	let total = 0;
	for (const seq of seqs) total += bySeq.get(seq) ?? 0;
	return total;
}
//#endregion
//#region src/tail.ts
/** Rough token estimate: chars / 4 (matches the dsh heuristic family). */
function approxTokenCount(text) {
	return Math.ceil(text.length / 4);
}
/** Port of `validate_summary`. */
function validateSummary(summary, template) {
	const trimmed = summary.trim();
	if (trimmed.length === 0 || trimmed === template.trim()) throw new Error("session memory summary is missing or still the template");
}
/** Port of `validate_tail_budget`. */
function validateTailBudget(tokens) {
	if (tokens > 4e4) throw new Error(`session memory raw tail exceeds ${MAX_RAW_TAIL_TOKENS} tokens`);
}
/** Port of `summary_budget_reminder`. */
function summaryBudgetReminder(summary) {
	const totalTokens = approxTokenCount(summary);
	const oversized = oversizedSections(summary);
	if (totalTokens <= 12e3 && oversized.length === 0) return "";
	let reminder = "";
	const overBudget = totalTokens > MAX_SESSION_MEMORY_TOTAL_TOKENS;
	if (overBudget) reminder += `\n\nCRITICAL: The session memory file is currently ~${totalTokens} tokens, which exceeds the maximum of ${MAX_SESSION_MEMORY_TOTAL_TOKENS} tokens. You MUST condense the file to fit within this budget. Aggressively shorten oversized sections by removing less important details, merging related items, and summarizing older entries. Prioritize keeping "Current State" and "Errors & Corrections" accurate and detailed.`;
	if (oversized.length > 0) reminder += `\n\n${overBudget ? "Oversized sections to condense" : "IMPORTANT: The following sections exceed the per-section limit and MUST be condensed"}:\n${oversized.join("\n")}`;
	return reminder;
}
/** Port of `summarize_oversized_sections` + `collect_oversized_section`. */
function oversizedSections(summary) {
	const sections = [];
	let currentHeader = "";
	let currentLines = [];
	for (const line of summary.split("\n")) if (line.startsWith("#")) {
		collectOversizedSection(sections, currentHeader, currentLines);
		currentHeader = line;
		currentLines = [];
	} else currentLines.push(line);
	collectOversizedSection(sections, currentHeader, currentLines);
	return sections;
}
function collectOversizedSection(sections, header, lines) {
	if (header.length === 0) return;
	const tokens = approxTokenCount(lines.join("\n"));
	if (tokens > 2e3) sections.push(`- "${header}" is ~${tokens} tokens (limit: ${MAX_SESSION_MEMORY_SECTION_TOKENS})`);
}
/** Port of `truncate_summary_for_compact`: shrink to total budget. */
function truncateSummaryForCompact(summary) {
	let text = summary;
	let tokens = approxTokenCount(text);
	if (tokens <= 12e3) return {
		text,
		wasTruncated: false
	};
	for (let pass = 0; pass < 3 && tokens > 12e3; pass += 1) {
		const sections = splitSections(text);
		if (sections.length === 0) break;
		let largest = sections[0];
		for (const section of sections) if (section.bodyTokens > largest.bodyTokens) largest = section;
		if (largest.bodyTokens <= 2e3) break;
		const body = truncateBody(largest.body, Math.floor(MAX_SESSION_MEMORY_SECTION_TOKENS * .7));
		text = text.replace(largest.body, body);
		tokens = approxTokenCount(text);
	}
	if (tokens > 12e3) {
		const budget = MAX_SESSION_MEMORY_TOTAL_TOKENS * 4;
		text = text.slice(0, budget);
	}
	return {
		text,
		wasTruncated: true
	};
}
function splitSections(text) {
	const lines = text.split("\n");
	const sections = [];
	let currentHeader = "";
	let currentBody = [];
	for (const line of lines) if (line.startsWith("#")) {
		if (currentHeader.length > 0) {
			const body = currentBody.join("\n");
			sections.push({
				header: currentHeader,
				body,
				bodyTokens: approxTokenCount(body)
			});
		}
		currentHeader = line;
		currentBody = [];
	} else currentBody.push(line);
	if (currentHeader.length > 0) {
		const body = currentBody.join("\n");
		sections.push({
			header: currentHeader,
			body,
			bodyTokens: approxTokenCount(body)
		});
	}
	return sections;
}
function truncateBody(body, charBudget) {
	if (body.length <= charBudget) return body;
	const cut = body.slice(0, charBudget);
	const lastNewline = cut.lastIndexOf("\n");
	return `${cut.slice(0, lastNewline > 0 ? lastNewline : charBudget)}\n_[truncated]_`;
}
/** Port of `format_compact_summary` (analysis/summary tag cleanup). */
function formatCompactSummary(summary) {
	let formatted = summary;
	const analysisStart = formatted.indexOf("<analysis>");
	if (analysisStart >= 0) {
		const analysisEnd = formatted.indexOf("</analysis>", analysisStart);
		if (analysisEnd >= 0) formatted = formatted.slice(0, analysisStart) + formatted.slice(analysisEnd + 11);
	}
	const summaryStart = formatted.indexOf("<summary>");
	if (summaryStart >= 0) {
		const contentStart = summaryStart + 9;
		const summaryEnd = formatted.indexOf("</summary>", contentStart);
		if (summaryEnd >= 0) formatted = `${formatted.slice(0, summaryStart)}Summary:\n${formatted.slice(contentStart, summaryEnd).trim()}${formatted.slice(summaryEnd + 10)}`;
	}
	while (formatted.includes("\n\n\n")) formatted = formatted.replaceAll("\n\n\n", "\n\n");
	return formatted.trim();
}
/** Port of `format_session_memory_summary`. */
function formatSessionMemorySummary(summary, wasTruncated, transcriptPath, memoryPath) {
	const formattedSummary = formatCompactSummary(summary);
	let formatted = `${SUMMARY_FRAMING.header}\n\n${formattedSummary}`;
	if (transcriptPath !== void 0) formatted += SUMMARY_FRAMING.transcriptSuffix(transcriptPath);
	formatted += SUMMARY_FRAMING.verbatim;
	formatted += SUMMARY_FRAMING.resume;
	if (wasTruncated) formatted += SUMMARY_FRAMING.truncatedSuffix(memoryPath);
	return formatted;
}
/**
* Port of `raw_tail_after_summary_boundary` + `calculate_tail_start` +
* `adjust_start_to_preserve_pairs`.
*
* Returns the tail seqs retained verbatim after a compact, or `null` when the
* boundary cannot be located (fingerprint mismatch or missing summary state).
*/
function selectTail(session, nodes, state) {
	const surface = session.surface.nodes;
	const surfaceSet = new Set(surface);
	let floorIndex = 0;
	for (let index = surface.length - 1; index >= 0; index -= 1) {
		const event = nodes.find((node) => node.seq === surface[index])?.event;
		if (event !== void 0 && event.type === "compaction/summary") {
			floorIndex = index + 1;
			break;
		}
	}
	let startIndex;
	if (state.last_summary_seq === void 0) startIndex = surface.length;
	else {
		const boundaryIndex = surface.indexOf(state.last_summary_seq);
		if (boundaryIndex < 0) throw new Error("session memory boundary not found");
		const fingerprint = state.last_summary_fingerprint;
		if (fingerprint === void 0) throw new Error("session memory boundary fingerprint missing");
		if (nodeFingerprint(session, state.last_summary_seq) !== fingerprint) throw new Error("session memory boundary fingerprint mismatch");
		startIndex = boundaryIndex + 1;
	}
	let start = Math.max(startIndex, floorIndex);
	let tail = surface.slice(start);
	let tokens = estimateRangeTokens(nodes, tail);
	let textItems = countTextItems(session, tail);
	while (start > floorIndex && (tokens < 1e4 || textItems < 5)) {
		const candidateSeq = surface[start - 1];
		const candidateTokens = estimateRangeTokens(nodes, [candidateSeq]);
		if (tokens > 0 && tokens + candidateTokens > 4e4) break;
		start -= 1;
		tail = surface.slice(start);
		tokens += candidateTokens;
		textItems += countTextItems(session, [candidateSeq]);
	}
	const resultCallIds = /* @__PURE__ */ new Set();
	for (const seq of tail) {
		const event = nodes.find((node) => node.seq === seq)?.event;
		if (event !== void 0 && event.type === "tool/result") resultCallIds.add(event.data.message.content[0]?.toolCallId ?? "");
	}
	if (resultCallIds.size > 0) for (let index = start - 1; index >= floorIndex; index -= 1) {
		const event = nodes.find((node) => node.seq === surface[index])?.event;
		if (event !== void 0 && event.type === "tool/call" && resultCallIds.has(event.data.callId)) start = index;
	}
	tail = surface.slice(start);
	tokens = estimateRangeTokens(nodes, tail);
	validateTailBudget(tokens);
	const shadowedSeqs = surface.slice(0, start).filter((seq) => surfaceSet.has(seq));
	if (shadowedSeqs.length === 0) return null;
	return {
		tailSeqs: tail,
		shadowedSeqs,
		tokens
	};
}
//#endregion
//#region src/engine.ts
/**
* SessionMemoryEngine: the dsh `CompactionEngine` implementation backed by
* the per-session summary file. Port of astral-code's `try_compact` path:
* summary + verbatim tail replace the shadowed history head.
* @module dsh-session-memory/engine
*/
function inspectEntryState(events) {
	let openTurn = null;
	let openTurnKnown = false;
	let unmatchedStart = false;
	let compactionKnown = false;
	for (let index = events.length - 1; index >= 0; index -= 1) {
		const event = events[index];
		if (!compactionKnown) {
			if (event.type === "compaction/start") {
				unmatchedStart = true;
				compactionKnown = true;
			} else if (event.type === "compaction/end") compactionKnown = true;
		}
		if (!openTurnKnown) {
			if (event.type === "turn/start") {
				openTurn = event.data.turn;
				openTurnKnown = true;
			} else if (event.type === "turn/end") openTurnKnown = true;
		}
		if (openTurnKnown && compactionKnown) break;
	}
	return {
		openTurn,
		unmatchedCompactionStart: unmatchedStart
	};
}
var SessionMemoryEngine = class extends CompactionEngine {
	config;
	overflowRetries = /* @__PURE__ */ new WeakMap();
	overflowAgents = /* @__PURE__ */ new WeakMap();
	constructor(ctx, config = {}) {
		super(ctx);
		this.config = {
			thresholdRatio: .75,
			summaryTemplate: "",
			storeDir: ".dsh/session-memory",
			...config
		};
		this._registerAutomaticCompaction();
	}
	/**
	* Register automatic between-step pressure and model-request overflow
	* recovery, mirroring `dsh-compaction-basic`. `compactIfNeeded` stays
	* dynamically dispatched so subclass overrides are honored at event time.
	*/
	_registerAutomaticCompaction() {
		const { ctx } = this;
		const logResult = (result, trigger) => {
			ctx.logger.info(`session-memory compaction (${trigger}): shadowed ${result.shadowedSeqs.length} surface nodes (seqs ${result.shadowedRange.start}-${result.shadowedRange.end}, ~${result.shadowedTokenCount} tokens)`);
		};
		ctx.on("agent/pre-step", async ({ agent, signal }, next) => {
			if (!signal.aborted) try {
				const result = await this.compactIfNeeded(agent, "pressure", signal);
				if (result !== null) logResult(result, "step pressure");
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				ctx.logger.warn(`session-memory step compaction failed: ${message}; continuing the turn`);
			}
			return next();
		});
		ctx.on("agent/status", ({ agent, status }) => {
			if (status === "idle") this.overflowRetries.delete(agent);
		});
		ctx.on("session/event", (session, event) => {
			if (event.type !== "assistant/message") return;
			const agent = this.overflowAgents.get(session);
			if (agent !== void 0) this.overflowRetries.delete(agent);
		});
		ctx.on("agent/request-error", async ({ agent, failure, signal }, next) => {
			if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted) return next();
			this.overflowAgents.set(agent.session, agent);
			const retries = this.overflowRetries.get(agent) ?? 0;
			if (retries >= 3) return next();
			const generation = agent.session.surface.replaceGeneration;
			let result;
			try {
				result = await this.compactIfNeeded(agent, "context-overflow", signal);
			} catch (recoveryError) {
				const message = recoveryError instanceof Error ? recoveryError.message : String(recoveryError);
				ctx.logger.warn(`session-memory overflow compaction failed: ${message}`);
				return next();
			}
			if (result === null) return next();
			this.overflowRetries.set(agent, retries + 1);
			if (agent.session.surface.replaceGeneration === generation && result.shadowedSeqs.length === 0) this.overflowRetries.set(agent, retries + 3);
			logResult(result, "context overflow");
			return next();
		});
	}
	storeFor(session) {
		return new SessionMemoryStore(session.id, `${this.config.storeDir}/${session.id}`);
	}
	/** Resolve the latest routed provider/model, mirroring compaction-basic. */
	routedTarget(session) {
		const header = session.requestHeader();
		if (header === void 0) return void 0;
		const config = header.config;
		if (config === void 0 || config.provider.length === 0 || config.model.length === 0) return;
		return {
			provider: config.provider,
			model: config.model
		};
	}
	async compactIfNeeded(agent, trigger, signal) {
		const store = this.storeFor(agent.session);
		await store.ensure(this.config.summaryTemplate);
		let state;
		try {
			state = await store.waitForRunningExtraction();
		} catch {
			return null;
		}
		const measurement = this.ctx.tokenMeter.measure(agent.session);
		const nodes = surfaceNodes(agent.session, measurement);
		if (trigger === "pressure") {
			const target = this.routedTarget(agent.session);
			if (target === void 0) return null;
			const contextTokens = (await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)).context?.contextWindow;
			if (contextTokens === void 0) return null;
			const threshold = Math.floor(contextTokens * this.config.thresholdRatio);
			if (measurement.totalTokens < threshold) return null;
		}
		try {
			validateSummary(await store.readSummary(), this.config.summaryTemplate);
			const tail = selectTail(agent.session, nodes, state);
			if (tail === null) return null;
			if (tail.shadowedSeqs.length === 0) return null;
			const first = tail.shadowedSeqs[0];
			const last = tail.shadowedSeqs[tail.shadowedSeqs.length - 1];
			return await this.compactRegion(first, last, agent, signal);
		} catch (error) {
			if (error instanceof SummaryMissingError) return null;
			this.ctx.logger?.warn(`dsh-session-memory compact refused: ${String(error)}`);
			return null;
		}
	}
	async compactNow(agent, signal, sourceCommandId) {
		signal.throwIfAborted();
		try {
			return await agent.runMaintenance(async (agentSignal) => {
				const operationSignal = AbortSignal.any([agentSignal, signal]);
				operationSignal.throwIfAborted();
				const store = this.storeFor(agent.session);
				let state;
				try {
					state = await store.waitForRunningExtraction();
				} catch {
					return null;
				}
				const measurement = this.ctx.tokenMeter.measure(agent.session);
				const nodes = surfaceNodes(agent.session, measurement);
				validateSummary(await store.readSummary(), this.config.summaryTemplate);
				const tail = selectTail(agent.session, nodes, state);
				if (tail === null || tail.shadowedSeqs.length === 0) return null;
				const first = tail.shadowedSeqs[0];
				const last = tail.shadowedSeqs[tail.shadowedSeqs.length - 1];
				return await this.compactRegionInner(first, last, agent, null, sourceCommandId, operationSignal);
			});
		} catch (error) {
			throw new ManualCompactionError("busy", "session-memory manual compaction requires an idle agent", { cause: error });
		}
	}
	async compactRegion(start, end, agent, signal) {
		return this.compactRegionInner(start, end, agent, "current-turn", void 0, signal);
	}
	/** The transaction body, ported from compaction-basic's `compactSurfaceRegion`. */
	async compactRegionInner(start, end, agent, ownerMode, sourceCommandId, signal) {
		const session = agent.session;
		const entry = inspectEntryState(session.events);
		if (entry.unmatchedCompactionStart) throw new Error("session-memory compactRegion: a compaction is already active");
		let owner;
		if (ownerMode === null) {
			if (entry.openTurn !== null) throw new ManualCompactionError("busy", "manual compaction: the session already has an open turn");
			owner = null;
		} else {
			if (entry.openTurn === null) throw new Error("session-memory compactRegion: no open turn — automatic compaction events must be enclosed in a turn");
			owner = entry.openTurn;
		}
		const measurement = this.ctx.tokenMeter.measure(session);
		const nodes = surfaceNodes(session, measurement);
		const surface = session.surface.nodes;
		if (!surface.includes(start) || !surface.includes(end)) throw new Error("session-memory compactRegion: range missing from surface");
		const startIdx = surface.indexOf(start);
		const endIdx = surface.indexOf(end);
		if (startIdx < 0 || endIdx < startIdx) throw new Error("session-memory compactRegion: reversed range");
		const shadowedSeqs = surface.slice(startIdx, endIdx + 1);
		if (!toolPairingBalancedBefore(session, start)) throw new Error("session-memory compactRegion: unbalanced left edge");
		if (!toolPairingBalancedAfter(session, end)) throw new Error("session-memory compactRegion: unbalanced right edge");
		const compactionId = CompactionId(randomUUID());
		const lifecycle = {
			compactionId,
			...sourceCommandId === void 0 ? {} : { sourceCommandId },
			turn: owner
		};
		const startEvent = session.append("compaction/start", lifecycle);
		let failure;
		try {
			const store = this.storeFor(session);
			await store.ensure(this.config.summaryTemplate);
			const summary = await store.readSummary();
			validateSummary(summary, this.config.summaryTemplate);
			const { text, wasTruncated } = truncateSummaryForCompact(summary);
			const formatted = formatSessionMemorySummary(text, wasTruncated, this.config.transcriptPath, store.summaryPath);
			const checkpointMessage = createUserMessage({
				content: [{
					type: "text",
					text: formatted
				}],
				source: compactCheckpointSource(compactionId, sourceCommandId)
			});
			const shadowedTokenCount = estimateRangeTokens(nodes, shadowedSeqs);
			const framedTokens = this.ctx.tokenMeter.estimateMessage(checkpointMessage);
			if (framedTokens >= shadowedTokenCount) throw new Error(`summary is not smaller than the shadowed content (${framedTokens} estimated framed tokens >= ${shadowedTokenCount})`);
			const after = this.ctx.tokenMeter.measure(session);
			if (JSON.stringify(after.nodes) !== JSON.stringify(measurement.nodes)) throw new Error("session-memory compactRegion: surface changed during summary read");
			const target = this.routedTarget(session);
			const summaryEvent = session.append("compaction/summary", {
				compactionId,
				...sourceCommandId === void 0 ? {} : { sourceCommandId },
				summary: [{
					type: "text",
					text: formatted
				}],
				shadowedRange: {
					start,
					end
				},
				shadowedSeqs: [...shadowedSeqs],
				shadowedTokenCount,
				provider: target?.provider ?? "session-memory",
				model: target?.model ?? "summary.md"
			});
			session.append("user/message", checkpointMessage, {
				surfaceOp: {
					op: "replace",
					start,
					end
				},
				sourceEventSeqs: [
					startEvent.seq,
					summaryEvent.seq,
					...shadowedSeqs
				]
			});
			const endEvent = session.append("compaction/end", {
				compactionId,
				...sourceCommandId === void 0 ? {} : { sourceCommandId },
				turn: owner
			});
			const fresh = this.ctx.tokenMeter.measure(session);
			await store.recordPostCompactBaseline(fresh.totalTokens, countToolCalls(session));
			return {
				compactionId,
				...sourceCommandId === void 0 ? {} : { sourceCommandId },
				startSeq: startEvent.seq,
				summarySeq: summaryEvent.seq,
				endSeq: endEvent.seq,
				summary: [{
					type: "text",
					text: formatted
				}],
				shadowedRange: {
					start,
					end
				},
				shadowedSeqs: [...shadowedSeqs],
				shadowedTokenCount
			};
		} catch (error) {
			failure = error instanceof Error ? error.message : String(error);
			session.append("compaction/end", {
				compactionId,
				...sourceCommandId === void 0 ? {} : { sourceCommandId },
				turn: owner,
				error: failure
			});
			throw error;
		}
	}
};
//#endregion
//#region src/sidechain.ts
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
const EDIT_TOOL_NAME = "edit";
/** Port of the sidechain-visible tool surface: Edit only. */
const EDIT_TOOL_SCHEMA = {
	name: EDIT_TOOL_NAME,
	description: "Edit the session notes file. Replaces one exact old_string occurrence with new_string.",
	parameters: {
		type: "object",
		properties: {
			file_path: {
				type: "string",
				description: "The absolute path of the notes file to edit."
			},
			old_string: {
				type: "string",
				description: "The exact text to replace. Must match exactly once in the file."
			},
			new_string: {
				type: "string",
				description: "The replacement text."
			}
		},
		required: [
			"file_path",
			"old_string",
			"new_string"
		],
		additionalProperties: false
	}
};
/** Port of `apply_summary_edit`: str_replace against the working text. */
function applySummaryEdit(currentText, args, summaryPath) {
	const filePath = typeof args.file_path === "string" ? args.file_path : "";
	const oldString = typeof args.old_string === "string" ? args.old_string : "";
	const newString = typeof args.new_string === "string" ? args.new_string : "";
	if (filePath.trim() === summaryPath.trim()) {}
	if (oldString.length === 0) return {
		text: currentText,
		result: "Error: old_string must be non-empty.",
		edited: false
	};
	const occurrences = currentText.split(oldString).length - 1;
	if (occurrences === 0) return {
		text: currentText,
		result: "Error: old_string was not found in the file.",
		edited: false
	};
	if (occurrences > 1) return {
		text: currentText,
		result: `Error: old_string occurs ${occurrences} times — provide more context to make it unique.`,
		edited: false
	};
	return {
		text: currentText.replace(oldString, newString),
		result: "Edit applied.",
		edited: true
	};
}
/** Port of `updater_prompt` + `substitute_prompt_variables`. */
function buildUpdaterPrompt(template, summaryPath, currentSummary, budgetReminder) {
	let output = "";
	let rest = template;
	for (;;) {
		const start = rest.indexOf("{{");
		if (start < 0) {
			output += rest;
			break;
		}
		output += rest.slice(0, start);
		const afterOpen = rest.slice(start + 2);
		const end = afterOpen.indexOf("}}");
		if (end < 0) {
			output += rest.slice(start);
			break;
		}
		const key = afterOpen.slice(0, end);
		if (key === "currentNotes") output += currentSummary;
		else if (key === "notesPath") output += summaryPath;
		else output += `{{${key}}}`;
		rest = afterOpen.slice(end + 2);
	}
	return `${output}${budgetReminder}`;
}
/** Port of `run_extraction` + `run_extraction_inner`. */
async function runExtraction(services, session, store, updatePromptTemplate, options, boundary) {
	const currentSummary = await store.readSummary();
	let workingText = currentSummary;
	const budgetReminder = summaryBudgetReminder(currentSummary);
	const updaterPrompt = buildUpdaterPrompt(updatePromptTemplate, store.summaryPath, currentSummary, budgetReminder);
	const assembly = await services.systemPrompt.assemble(assembleContextFor(options.agent, options.signal));
	const system = renderPrompt(assembly);
	const messages = [...session.deriveMessages(), createUserMessage({
		content: [{
			type: "text",
			text: updaterPrompt
		}],
		source: {
			kind: "plugin",
			plugin: "dsh-session-memory"
		}
	})];
	let edited = false;
	let round = 0;
	for (;;) {
		round += 1;
		const assembler = new BlockAssembler();
		const stream = services.llm.stream({
			provider: options.provider,
			model: options.model,
			system,
			messages,
			tools: [EDIT_TOOL_SCHEMA],
			purpose: "compaction",
			signal: options.signal
		});
		for await (const chunk of stream) assembler.push(chunk);
		const blocks = assembler.blocks();
		if (process.env.DSH_SESSION_MEMORY_DEBUG !== void 0) console.error(`[dsh-session-memory] sidechain round ${round} blocks: ${JSON.stringify(blocks)}`);
		const assistantBlocks = [];
		let anyToolCall = false;
		for (const block of blocks) {
			if (block.type !== "tool-call") continue;
			anyToolCall = true;
			assistantBlocks.push(block);
			if (block.name === EDIT_TOOL_NAME) {
				let args = {};
				try {
					args = JSON.parse(block.arguments);
				} catch {
					args = {};
				}
				const outcome = applySummaryEdit(workingText, args, store.summaryPath);
				workingText = outcome.text;
				assistantBlocks.push({
					type: "tool-result",
					toolCallId: block.id,
					content: [{
						type: "text",
						text: outcome.result
					}]
				});
				if (outcome.edited) edited = true;
			} else assistantBlocks.push({
				type: "tool-result",
				toolCallId: block.id,
				content: [{
					type: "text",
					text: DENY_TOOL_MESSAGE(store.summaryPath)
				}],
				isError: true
			});
		}
		if (!anyToolCall) break;
		if (round >= options.maxRounds) throw new Error("session memory extraction exceeded tool-call rounds");
		messages.push(createAssistantMessageWithBlocks(assistantBlocks, options));
	}
	if (edited) await store.atomicWriteSummary(workingText);
	await finishExtractionSuccess(store, boundary);
	return boundary;
}
async function finishExtractionSuccess(store, boundary) {
	await store.finishExtraction({
		seq: boundary.seq,
		fingerprint: boundary.fingerprint,
		tokens: boundary.tokens,
		toolCalls: boundary.toolCalls
	});
}
function createAssistantMessageWithBlocks(blocks, options) {
	return createMessage({
		role: "assistant",
		content: blocks,
		source: {
			kind: "model",
			provider: options.provider,
			model: options.model
		}
	});
}
//#endregion
//#region src/index.ts
const Config = z.object({
	storeDir: z.string().default(".dsh/session-memory"),
	summaryTemplate: z.string().default(DEFAULT_SUMMARY),
	updatePrompt: z.string().default(DEFAULT_UPDATE_PROMPT),
	thresholdRatio: z.number().default(.75),
	initMessageTokens: z.number().default(DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT),
	updateTokenInterval: z.number().default(DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE),
	updateToolCallInterval: z.number().default(10),
	sidechainProvider: z.string().default(""),
	sidechainModel: z.string().default(""),
	transcriptPath: z.string().default("")
});
const inject = [
	"llm",
	"tokenMeter",
	"systemPrompt"
];
function apply(ctx, config = {}) {
	const options = {
		storeDir: ".dsh/session-memory",
		summaryTemplate: DEFAULT_SUMMARY,
		updatePrompt: DEFAULT_UPDATE_PROMPT,
		thresholdRatio: .75,
		initMessageTokens: DEFAULT_MINIMUM_MESSAGE_TOKENS_TO_INIT,
		updateTokenInterval: DEFAULT_MINIMUM_TOKENS_BETWEEN_UPDATE,
		updateToolCallInterval: 10,
		sidechainProvider: "",
		sidechainModel: "",
		transcriptPath: "",
		...config
	};
	ctx.plugin(SessionMemoryEngine, options);
	const services = {
		llm: ctx.llm,
		systemPrompt: ctx.systemPrompt
	};
	const logger = ctx.logger;
	const tokenMeter = ctx.tokenMeter;
	const sessions = /* @__PURE__ */ new WeakMap();
	const runningExtractions = /* @__PURE__ */ new Set();
	ctx.on("agent/pre-step", ({ agent }, next) => {
		sessions.set(agent.session, agent);
		return next();
	});
	const maybeSpawnExtraction = async (session) => {
		if (runningExtractions.has(session)) return;
		const agent = sessions.get(session);
		if (agent === void 0) return;
		const store = new SessionMemoryStore(session.id, `${options.storeDir}/${session.id}`);
		await store.ensure(options.summaryTemplate);
		spawnExtraction(services, tokenMeter, logger, session, agent, store, options, runningExtractions);
	};
	ctx.on("session/event", (session, event) => {
		if (event.type === "turn/end") queueMicrotask(() => void maybeSpawnExtraction(session));
	});
	ctx.on("agent/status", ({ agent, status }) => {
		if (status === "idle") maybeSpawnExtraction(agent.session);
	});
}
async function spawnExtraction(services, tokenMeter, logger, session, agent, store, options, runningExtractions) {
	runningExtractions.add(session);
	try {
		const state = await store.readState();
		const measurement = tokenMeter.measure(session);
		const tokens = measurement.totalTokens;
		const toolCalls = countToolCalls(session);
		const lastSummaryTokens = state.last_summary_tokens ?? 0;
		const lastSummaryToolCalls = state.last_summary_tool_calls ?? 0;
		if (state.last_summary_seq === void 0 ? tokens < options.initMessageTokens : tokens - lastSummaryTokens < options.updateTokenInterval && toolCalls - lastSummaryToolCalls < options.updateToolCallInterval) return;
		if (!lastSurfaceEventIsNaturalBreak(session)) return;
		const surface = session.surface.nodes;
		if (surface.length === 0) return;
		const boundarySeq = surface[surface.length - 1];
		const fingerprint = nodeFingerprint(session, boundarySeq);
		if (fingerprint === void 0) return;
		const boundary = {
			seq: boundarySeq,
			fingerprint,
			tokens: surfaceNodes(session, measurement).filter((node) => node.seq === boundarySeq).reduce((sum, node) => sum + node.tokens, 0),
			toolCalls
		};
		const target = resolveSidechainTarget(session, options);
		if (target === void 0) {
			await store.finishExtraction(void 0, "no provider/model route for sidechain extraction");
			return;
		}
		await store.markExtractionStarted();
		try {
			await runExtraction(services, session, store, options.updatePrompt, {
				agent,
				provider: target.provider,
				model: target.model,
				maxRounds: 6
			}, boundary);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			await store.finishExtraction(void 0, message);
			logger.warn(`session-memory extraction failed: ${message}`);
		}
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		logger.warn(`session-memory extraction spawn failed: ${message}`);
	} finally {
		runningExtractions.delete(session);
	}
}
function resolveSidechainTarget(session, options) {
	if (options.sidechainProvider.length > 0 && options.sidechainModel.length > 0) return {
		provider: options.sidechainProvider,
		model: options.sidechainModel
	};
	const config = session.requestHeader()?.config;
	if (config !== void 0 && config.provider.length > 0 && config.model.length > 0) return {
		provider: config.provider,
		model: config.model
	};
}
//#endregion
export { Config, apply, inject };
