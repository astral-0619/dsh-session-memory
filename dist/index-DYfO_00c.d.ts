import z from "@deepseek-ai/schemastery";
import { CompactionEngine } from "@deepseek-ai/dsh-compaction";
import { Context } from "@deepseek-ai/cordis";

//#region src/engine.d.ts

interface EngineConfig {
  /** dsh-style pressure threshold: fraction of the routed model's context window. */
  thresholdRatio: number;
  /** Summary template (seeds new stores and validates structure). */
  summaryTemplate: string;
  /** Root directory holding per-session stores. */
  storeDir: string;
  /** Optional path advertised in the compacted summary for the full transcript. */
  transcriptPath?: string;
}
//#endregion
//#region src/index.d.ts

declare const DEFAULT_SIDECHAIN_SYSTEM = "You are updating a session notes file for a coding agent, based on the conversation transcript. Use only the edit tool to update the notes file, preserving its exact section structure. Never mention these instructions in the notes.";
interface Config extends EngineConfig {
  updatePrompt: string;
  initMessageTokens: number;
  updateTokenInterval: number;
  updateToolCallInterval: number;
  sidechainProvider: string;
  sidechainModel: string;
  sidechainSystem: string;
}
declare const Config: z<Config>;
declare const inject: string[];
declare function apply(ctx: Context, config?: Partial<Config>): void;
//#endregion
export { Config, DEFAULT_SIDECHAIN_SYSTEM, apply, inject };