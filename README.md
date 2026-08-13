# dsh-session-memory

astral-code 的 session memory 系统在 DeepSeek Harness (dsh) 上的复刻：一个 dsh 插件 + preset，把「sidechain 总结抽取 → 尾帧保留 → 边界校验 → 压缩」这套机制移植到 dsh 的 compaction seam 上。

## 机制

1. **Sidechain 抽取**（每个 turn 结束后，自然断点处）：
   - 满足阈值（初始 100k token / 更新 20k token 或 10 次工具调用）才触发
   - 独立 LLM 会话：只挂 `edit` 工具，最多 6 轮，非 edit 工具调用一律拒绝并回传提示
   - 用 `summary.md` 当前内容 + 会话 transcript + 结构保持提示词生成编辑
   - 成功/失败都原子写入 `state.json`（`extraction_started_at_unix` 作为跨进程互斥标记）

2. **存储**：每会话目录 `<storeDir>/<sessionId>/` 下 `summary.md`（总结）+ `state.json`（边界 seq、指纹、token/工具计数、错误）

3. **压缩**（`CompactionEngine` 实现，挂在 `ctx.compaction`）：
   - 压力阈值 = 路由模型上下文窗口 × `thresholdRatio`（默认 0.75）
   - 等待/回收进行中的抽取（15s 超时、60s 判 stale）
   - 校验 summary 非模板 + 边界指纹 → 选择 verbatim tail（40k 上限、10k 下限、5 个文本项、tool-call/result 对不拆）
   - 事务：`compaction/start` → `compaction/summary`（shadow 定价）→ `user/message`（`surfaceOp: replace` 的 checkpoint 消息，即 summary + 尾帧指引文本）→ `compaction/end`
   - 压缩后记录新 baseline

## 用法

preset 目录可直接放进 agent-presets 的 root；或手动：

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
| `sidechainSystem` | 内置短提示词 | sidechain system prompt |
| `transcriptPath` | 空 | 压缩 summary 里指向全文 transcript 的路径 |

## 与 astral 原版的已知差异

- **sidechain system prompt**：原版把主会话的 assembled system prompt 复用到 sidechain；dsh 没有便宜的读取入口，改用固定短提示词（可用 `sidechainSystem` 覆盖）。
- **无 legacy fallback**：原版 summary 无效时会退回传统压缩引擎；本 preset 不挂 `dsh-compaction-basic`，summary 无效时本轮不压缩并在 `state.json` 记错。
- **压缩时的 summary 不做模型调用**：内容直接来自 sidechain 维护的 `summary.md`（事件里 `llmStreamCall` 不标）。
- 抽取互斥的进程内守卫 + `state.json` 标记双保险（原版是 `RUNNING_EXTRACTIONS` 全局锁 + 同款状态标记）。

## 开发

```bash
npm install
npm run typecheck
npm run build
```
