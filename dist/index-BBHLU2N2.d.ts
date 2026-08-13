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

interface Config extends EngineConfig {
  updatePrompt: string;
  initMessageTokens: number;
  updateTokenInterval: number;
  updateToolCallInterval: number;
  sidechainProvider: string;
  sidechainModel: string;
}
declare const Config: z<Config>;
declare const inject: string[];
declare function apply(ctx: Context, config?: Partial<Config>): void;
//#endregion
export { Config, apply, inject };