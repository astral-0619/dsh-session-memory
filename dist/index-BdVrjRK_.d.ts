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
  /**
   * Await the sidechain extraction inside the turn/end listener. One-shot
   * drivers (dsh headless) exit the process at quiescence, tearing down the
   * LLM adapter registry before background extraction can run; awaiting keeps
   * the extraction inside the turn. Long-lived harnesses leave this false so
   * extraction stays background, like the original.
   */
  awaitOnTurnEnd: boolean;
}
declare const Config: z<Config>;
declare const inject: string[];
declare function apply(ctx: Context, config?: Partial<Config>): void;
//#endregion
export { Config, apply, inject };