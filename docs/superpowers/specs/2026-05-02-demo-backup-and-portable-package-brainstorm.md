# Demo Backup & Portable Package — Brainstorm Log

**Date**: 2026-05-02
**Owner**: Ed Chi
**Project**: habit-memory-demo (PC Simulator)
**Goal**: 为 system demo 做好 backup —— LLM 故障、网络故障、机器故障等 contingency；同时产出可迁移的打包文档（环境/依赖/配置/seed 数据）。

> 规则（feedback memory）：每一轮 Q&A 立即落档，防 compact 丢失。

---

## 项目现状速记（Round 0 探查）

- 入口：`streamlit run simulator/app.py`（Tab1 学习 + Tab2 推荐）
- Runtime 强外网依赖：
  - OpenAI Chat (`gpt-4.1`) — `panoramix_core/llm_client.py`
  - OpenAI Embedding (`text-embedding-ada-002`) — `panoramix_core/embedder.py`
- 第三方依赖（grep 出来的）：`streamlit, openai, chromadb, pydantic, numpy, matplotlib, requests, jwt, pyyaml, pytest`
- 持久化：
  - SQLite `storage/habit_memory.db`（gitignore）
  - ChromaDB `storage/memories/`（gitignore）
- 视频/Comfy/Kling 管线：理解为**离线预渲染**，demo 现场不依赖
- **缺失的工程化产物**：`requirements.txt`、`pyproject.toml`、`README.md`、`.env.example` 都没有

## 用户初始诉求

1. **OpenAI 被封** → 已申请 OpenRouter API key 作为备选
2. **现场网络差** → 需要预案
3. 让我补充其他可能的故障模式
4. 输出"打包迁移所需的环境配置 + 各种要求"文档

## 我准备问的清单（按优先级）

1. Demo 范围：Tab1 / Tab2 / 两者都演？哪些是 live、哪些是录屏/seed
2. Demo 机器：是当前 dev box，还是另一台？操作系统？
3. 网络条件：可拉热点 / 完全离线 / 仅会场 Wi-Fi？
4. Live LLM 调用 vs 预生成 cache：是否接受"现场零 LLM 调用"
5. Seed 数据：要不要把演示用的 user profile + ChromaDB + SQLite 一起 freeze 进迁移包
6. 备援机器：有没有第二台笔记本可作 hot standby
7. 录屏兜底：如果一切失败，是否准备一段全程录屏作为最后保险
8. 时间窗：距离 demo 还有多少天，决定我能做多少工程化

---

## Round 1 — Q

**问题**：先定 demo 范围。下面 4 选 1（影响后面所有 backup 设计）：

- **A. 仅 Tab1（学习阶段）**：现场喂信号 → 触发 DBSCAN → 生成 Habit 卡，需要 live LLM
- **B. 仅 Tab2（推荐阶段）**：基于 pre-seeded user profile，现场 trigger 推荐卡，可以做到"零 LLM 调用"
- **C. Tab1 + Tab2 双演**：完整闭环，最有冲击力但 LLM/网络风险最大
- **D. Tab1 + Tab2 + 视频片段**：再串入 day1 的几段离线渲染好的视频做 storytelling

## Round 1 — A

- **范围 = D**：Tab1 + Tab2 + 视频片段。视频离线串入已经做完，不再大改。
- **观众**：雷诺（Renault）员工，零散反复多轮演示
- **时长**：5–10 分钟，尽量压到 5 分钟
- **时间表**：距离 demo 三周；**最后一周需要去法国做迁移部署**
- 当前流程已按现有开发结果设计好

## Round 1 — Insights（不再追问）

1. **"反复多轮"是关键约束** — 不是一次性 demo，而是同一台机器一天内被多组人员看 N 次。所以 backup 必须支持**幂等重置 / 一键回到干净起点**，且 LLM 调用要做缓存（避免重复花钱 + 延迟）。
2. **法国部署最后 1 周** = 我手上只有 **2 周**做工程化，第 3 周是适配/救火窗口。任何 backup 必须在第 2 周末"可装箱"。
3. **D 方案的 LLM 触点** 集中在 Tab1（habit detection LLM 调用 + embedding）和 Tab2（推荐生成）。视频纯本地播放，不算风险。
4. **跨国部署** = 时区差异（法国上班、美国/中国凌晨），现场出问题不一定能远程救场，所以备份方案必须**离线可独立跑通**。
5. OpenRouter fallback 已有，但还需要：API quota 备案、二号 OpenAI key、本地 embedding 模型 fallback（避免封号一锅端）。

---

## Round 2 — Q

**问题**：现场机器和网络条件，下面 6 道，请逐条 1–2 句话回答（这一轮不分 A/B 选项，因为太具体）：

1. **演示机器**：是带你这台 dev Linux 笔记本去法国，还是雷诺现场提供机器/我们要新装一台？
2. **目标 OS**：如果是新机器，是 Ubuntu/Mac/Windows？
3. **现场网络**：雷诺会场是公司内网（可能限制 OpenAI 域名）/ 公共 Wi-Fi / 自己 4G 热点？有没有"完全离线"这种最坏情况要兜底？
4. **能否携带二号机**：现场是否会有第二台笔记本作为 hot standby（同步好的镜像，主机崩了 30 秒切过去）？
5. **GPU**：演示机要不要 GPU？（影响是否能跑本地 embedding fallback，比如 BGE / E5）
6. **录屏兜底**：是否接受"如果当场技术全炸，切到一段完整流程的录屏 + 解说"作为最后保险？

回完这 6 条我就能开始出 2–3 个 backup 方案给你选。

## Round 2 — A

1. 源机：本地 **Linux 台式机**（dev）
2. 目标机：另一台 **Linux 笔记本**（Ubuntu）→ 带去法国
3. 现场网络：**公共 Wi-Fi**（无完全离线硬要求）
4. 没有第二台笔记本；**但本地台式机可架远程服务，演示机可远程访问**
5. 演示笔记本 **GPU 不确定，最坏情况是没有/很差**
6. 录屏：接受作为兜底，**用户自己录**，但要写入最终文档提醒不要忘记

## Round 2 — Insights（不再追问）

1. **"本地台式机当远程兜底" 是个很强的牌**：可以做 Tailscale / Cloudflare Tunnel，让演示笔记本把"OpenAI 调用"路由到家里台式机的 LLM 代理 —— 等于自带一个"私有 LLM 镜像"。如果现场 Wi-Fi 直连 OpenAI 超时但能上 Tailscale，就能救命。
2. **公共 Wi-Fi 是高风险**：DNS 污染、HTTPS 中断、限流都会发生。所有外部调用必须有 **超时 + 重试 + 多 provider fallback**。
3. **没 GPU → 本地 embedding fallback 必须 CPU 友好**：放弃 BGE-large 之类，改用 **预计算 embedding cache**（演示场景固定，提前把所有可能的 fact 文本算好 embedding 存进 sqlite/json）。
4. **"反复多轮"+"5 分钟" + "无 GPU" → LLM 缓存是强制的**：第一轮演示填充缓存，后续 N 轮全部击中缓存，做到 0 次外部调用。
5. **录屏兜底独立保管**：必须在最终交付文档里专门列一条 checklist：「确认 mp4 已拷贝至演示笔记本本地 + U 盘」。用户自己会录，我只负责提醒。

---

## Round 3 — Q

最后一轮"输入条件"问题，3 道：

1. **现场交互模式**：演示是**脚本化的固定路径**（同一组信号、同一个 user，每次跑出来都一样），还是**允许观众现场输入新数据**（比如让雷诺员工自己点选信号）？这直接决定能否做"全缓存零 LLM"兜底。
2. **Seed 数据策略**：你倾向哪种？
   - (a) 演示包里**预置一份 frozen 的 ChromaDB + SQLite**（user profile 已学好），现场从 Tab2 开始演
   - (b) 现场从 Tab1 开始喂信号现学，**信号脚本固定**，命中本地 LLM 缓存即返回
   - (c) 两者都要，按需切换
3. **远程台式机的可达性**：你这台 dev 台式机是**24 小时开机**吗？演示期间（法国白天 = 你这边凌晨）能不能保证它是开着、可以接 Tailscale/Cloudflare Tunnel 的状态？

回完这 3 条，下一条消息我就直接出 **2–3 个 backup 方案 + 推荐项**。

## Round 3 — A

1. **脚本化固定流程**为主；偶尔**自己**（不是观众）现场输入新数据
2. Seed 策略：**(b)** —— 从 Tab1 开始喂固定信号现学，命中本地 LLM 缓存即返回
3. 本地台式机 **24h 开机**；不熟 Tailscale，但之前用过"把 localhost 投射成域名"的工具（即 ngrok / cloudflared / localtunnel 一类）

## Round 3 — Insights（不再追问）

1. **脚本化 + 自学(b) → 全 LLM 缓存可行性 100%**：固定信号脚本 → 固定 prompt → 固定 hash → 全部预 warm 进缓存。第 1 轮进现场之前在酒店预演一次，所有调用就被永久缓存。
2. **"自己偶尔输新数据" 是 escape hatch**：这条路径不能上 cache（prompt 不可预测），但只在 Q&A 阶段用，所以可走 OpenRouter / 远程隧道 fallback，失败也不影响主线 demo。
3. **远程隧道选型**：因为用户熟 ngrok 心智模型，方案直接说"用 cloudflared 把家里台式机的 LLM 代理服务投射成 https URL"，不引入 Tailscale 这种新概念。
4. **本地代理服务长什么样**：在你台式机上跑一个简单的 Flask/FastAPI 服务，路由 `/v1/chat/completions` 转发到 OpenAI（等价于一个 thin proxy），cloudflared 投射成 HTTPS。演示笔记本把 `OPENAI_BASE_URL` 指过去就走家里的网络出口（你家里能上 OpenAI = 法国能上 OpenAI）。
5. **缓存层语义**：键 = `sha256(model + prompt + temperature)`，值 = response 文本。文件 `storage/llm_cache.json`，伴随仓库一起迁移。Embedding 同理走另一份缓存 `storage/embedding_cache.json`。

---

## Approaches — 2 个方案 + 我的推荐

### 用户选择：方案 A，但调整调用链顺序

用户决策（Round 4）：**Cache 不是优先，是兜底**。优先级链调整为：

```
OpenAI 主 key（超时 5s）
   │ fail
   ▼
OpenRouter（同模型映射，5s）
   │ fail
   ▼
家里台式机 cloudflared 隧道 LLM proxy（借家里出口走 OpenAI）
   │ fail
   ▼
本地 LLM cache（最后兜底，replay 历史响应）
   │ miss
   ▼
明确报错 + UI 提示「降级到录屏」
```

**理由（用户）**：网络好的时候要展示真实 live LLM 能力，cache 只在三条网络通路都断的时候出场。

### Trade-off 注释（待用户确认是否接受）

- **代价 1（成本）**：反复多轮演示，每轮都打真实 OpenAI ≈ N × token 费用。脚本化路径用 cache 几乎免费。
- **代价 2（确定性）**：即使 temperature=0.2，多次调用返回文本会有微小差异。10 轮演示给雷诺不同员工看，每次输出可能不同 —— 有人会问"为什么这次和上次说的不一样"。
- **代价 3（延迟）**：公共 Wi-Fi 打 OpenAI 国际线路，2–5 秒等待是常态；cache 是 0ms。
- **缓解**：保留一个**紧急 cache-only 开关**（环境变量 `DEMO_FORCE_CACHE=1` 或 sidebar 一个按钮），默认关闭，万一现场网络抖动可一键切到全 cache 模式。这样既保留 live-first 体验，又有兜底逃生口。

### Round 4 — 用户确认 + 隐蔽开关需求

- 用户接受 trade-off 1/2/3 全部
- 紧急 cache 开关**必须有**，但**位置要隐蔽**，对观众不可见，对我们可发现
- **采纳方案**：URL query param `?cache_only=1` 切换 + sidebar 底部一行小灰字 indicator（`live ✓` / `CACHE ⚠`）作为状态显示。观众扫一眼看不出，演示者一眼就知道当前在哪个模式。

### Round 5 — Section 1 architecture sign-off

- 用户回复「看不懂，但 ok」 → 接受 Section 1 全部技术细节
- 衍生 feedback memory：用户对实现细节信任 dev lead 拍板，后续 section 用更人话的语气
- 已存储 memory: `feedback_trust_implementation_details.md`

### Round 7 — Timeout 调查与重设

**用户提醒**：5s timeout 太激进，公共 Wi-Fi 国际打 OpenAI 会误杀；用户记忆中是 30s 但要求复核。

**实际查证**（grep 全代码）：

- `panoramix_core/llm_client.py:29`：`OpenAI()` 没设 timeout，吃 SDK 默认值 **connect 5s + read 600s（10 分钟！）**
- `panoramix_core/embedder.py`：同样没设
- 唯一显式：`scripts/video_gen/pipeline.py:68` 设了 120s（视频管线，跟 demo 无关）

**结论**：现在如果 OpenAI 不响应，demo 卡 10 分钟才报错，是潜在大坑，**必须修**。

**重设 timeout（待用户确认）**：

| 节点 | connect | read | 总上限 |
|---|---|---|---|
| OpenAI 直连 | 5s | 15s | 20s |
| OpenRouter | 5s | 15s | 20s |
| 家里隧道 | 5s | 10s | 15s |
| Cache | — | — | <1ms |

最坏累计 55s 走完三路；典型 2–5s 即返回。

**衍生待办**：除了新加路由器层，还要顺手修 `llm_client.py` 和 `embedder.py` 的 OpenAI client 显式 timeout。

**用户决策**：选 (a) 15s/15s/10s，到现场再测调。

### Round 8 — 用户更正调用链顺序假设

**我的错误建议**：Section 4 #4 我说"建议在法国直接用隧道当主路，避开 OpenAI 风控"。

**用户更正**：法国 IP 调 OpenAI 是低风险（OpenAI 正常服务），国内家里 IP 调 OpenAI 反而高风险（OpenAI 不正式服务中国大陆），把隧道当主路是把更脆的环节前置，错误。

**最终决策**：调用链顺序不变 = `OpenAI(法国直连) → OpenRouter → 隧道(国内出口) → Cache`。

**Risk Register 更新**：
- #4 法国 IP 风控：降级 🟢，不主动处理
- 新增 #4'：**隧道本身就是脆弱节点**（家里 IP 反而更容易被 OpenAI 封），所以隧道仅作"OpenAI + OpenRouter 都断"的双重失败兜底，不当首选

**衍生 memory**：保存"国内家里 IP 调 OpenAI 比法国 IP 更脆弱"这条领域知识，避免未来再犯类似假设。

### Round 9 — 时间表 sign-off

- 三周划分：接受
- Dry-run 第二台机器：用户有借机条件，确认
- P2 任务：**Week 2 必须全部完成**，不延后到 Week 3
- 整体节奏：尽早开工

→ Brainstorm 阶段结束，进入 spec 落档 + 自检 + 用户审阅。



### Round 6 — Section 2 关键决策：U 盘整包拷贝

- **不走 git**（包括 git LFS / Release asset 全部不要）
- 迁移单元 = **一个完整文件夹** + U 盘物理拷贝到法国
- 衍生需求：
  - 必须包含 `wheels/` 目录（预下载所有 pip 包），新笔记本可**离线 pip install**，避免酒店/会场 Wi-Fi 卡 pypi
  - frozen 数据（fixtures、视频、cache）直接以文件形式放在包内，无需任何同步工具
  - Makefile / setup.sh 还在，只是定位变成"包内自带的脚本"
  - 不需要 `.gitignore` 调整、不需要决定 LFS



