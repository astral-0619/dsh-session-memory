# dsh-session-memory

astral-code 的 session memory 系统在 DeepSeek Harness (dsh) 上的复刻：一个 dsh 插件 + host-plane patch 片段，把「sidechain 总结抽取 → 尾帧保留 → 边界校验 → 压缩」这套机制移植到 dsh 的 compaction seam 上。

## 机制

1. **Sidechain 抽取**（每个 turn 结束后，自然断点处）：
   - 满足阈值（初始 100k token / 更新 20k token 或 10 次工具调用）才触发
   - 触发 seam：turn/end 后后台跑（长驻 harness）；同时挂 `session/flush` 监听并 await——headless 等一次性驱动在退出前会 await `sessions.flush()`，这样 sidechain 能在进程退出前跑完（实测：不挂 flush 监听的话，树 dispose 后 adapter 注册表被拆，抽取必失败）
   - 独立 LLM 会话，**上下文原样 fork 主会话**：system prompt 用主循环同款组装（`ctx.systemPrompt.assemble(assembleContextFor(agent))` + `renderPrompt`），messages 直接取 `session.deriveMessages()`，唯一差异是工具面只挂 `edit`，非 edit 工具调用一律拒绝并回传提示，最多 6 轮
   - updater 提示词（当前 summary + 结构保持要求）作为最后一条 user message 追加；edit 失败回传结果让模型重试，模型不再调工具时结束
   - 服务经 `ctx.get()` 取（不用 property proxy，避免 inject/inactive-context 坑）；成功/失败都原子写入 `state.json`（`extraction_started_at_unix` 作为跨进程互斥标记）

2. **存储**：每会话目录 `<storeDir>/<sessionId>/` 下 `summary.md`（总结）+ `state.json`（边界 seq、指纹、token/工具计数、错误）

3. **压缩**（`CompactionEngine` 实现，挂在 `ctx.compaction`）：
   - 压力阈值 = 路由模型上下文窗口 × `thresholdRatio`（默认 0.75）
   - 等待/回收进行中的抽取（15s 超时、60s 判 stale）
   - 校验 summary 非模板 + 边界指纹 → 选择 verbatim tail（40k 上限、10k 下限、5 个文本项、tool-call/result 对不拆）
   - 事务：`compaction/start` → `compaction/summary`（shadow 定价）→ `user/message`（`surfaceOp: replace` 的 checkpoint 消息，即 summary + 尾帧指引文本）→ `compaction/end`
   - 压缩后记录新 baseline

## 用法

`preset/cordis.patch.yml` 是 host-plane 补丁片段（compaction 引擎是进程级服务，**不能**作为 agent preset 挂载——dsh 的 agent preset 只允许 `isolate` realm 内的服务）。二选一：

- 粘进 profile 的 `cordis.patch.yml`
- 或 `--patch ./preset/cordis.patch.yml` 覆盖

该片段会 disable base 的 `compaction-basic` 行并插入 `session-memory` 行（两者不能同时注册 `ctx.compaction`）。

或手动（不用 loader）：

```ts
import { apply } from 'dsh-session-memory'
apply(ctx, { storeDir: '.dsh/session-memory', thresholdRatio: 0.75 })
```

`apply(ctx, config)` 会注册 `SessionMemoryEngine`（compaction service）+ turn 结束监听器。

## 配置

| key | 默认 | 说明 |
| --- | --- | --- |
| `storeDir` | `.dsh/session-memory` | 每会话存储根目录 |
| `summaryTemplate` | astral 同款模板 | 新会话的种子 summary |
| `updatePrompt` | astral 同款提示词 | sidechain 编辑指令 |
| `thresholdRatio` | `0.75` | 压缩压力阈值（× 路由模型上下文窗口） |
| `initMessageTokens` | `100000` | 首次抽取的 token 阈值 |
| `updateTokenInterval` | `20000` | 更新抽取的 token 间隔 |
| `updateToolCallInterval` | `10` | 更新抽取的工具调用间隔 |
| `sidechainProvider` / `sidechainModel` | 空（跟随路由） | sidechain LLM 路由覆盖 |
| `transcriptPath` | 空 | 压缩 summary 里指向全文 transcript 的路径 |

## 与 astral 原版的已知差异

- **sidechain system prompt 与原版一致**：通过 `ctx.systemPrompt.assemble(assembleContextFor(agent))` 取主循环同款组装并 `renderPrompt`，前缀不变（工具差异只在工具面，不进 system 文本）。
- **无 legacy fallback**：原版 summary 无效时会退回传统压缩引擎；本 patch 片段 disable 了 `dsh-compaction-basic`，summary 无效时本轮不压缩并在 `state.json` 记错。
- **压缩时的 summary 不做模型调用**：内容直接来自 sidechain 维护的 `summary.md`（事件里 `llmStreamCall` 不标）。
- 抽取互斥：进程内 per-session promise 守卫（flush 路径会 await 在跑的抽取）+ `state.json` 标记双保险（原版是 `RUNNING_EXTRACTIONS` 全局锁 + 同款状态标记）。

## 开发

```bash
npm install
npm run typecheck
npm run build
```

## 黑盒验证

用真 dsh headless profile + 真实 DeepSeek API（`DEEPSEEK_API_KEY`）跑过一次端到端：插件经 `dsh plugin --profile headless add <tarball>` 挂载，profile 的 `cordis.patch.yml` 用 `--patch` 片段 disable `compaction-basic` 并插入本插件；跑完一个任务后 `<storeDir>/<sessionId>/summary.md` 出现真实 LLM 生成的章节内容、`state.json` 记录边界（`last_summary_seq`/指纹/token 数）。侧链实际行为：round 1 出 reasoning + 多个 edit 调用、非 edit 工具调用被拒绝并记进 Errors & Corrections、edit 应用后逐轮继续直到模型停手。
