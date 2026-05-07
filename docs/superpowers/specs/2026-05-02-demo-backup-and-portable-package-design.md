# Demo Backup & Portable Package — Design Spec

**Date**: 2026-05-02
**Owner**: Ed Chi
**Project**: habit-memory-demo (PC Simulator)
**Demo Date**: 预计 2026-05-23 前后（最后一周飞法国适配）
**Audience**: 雷诺（Renault）员工，多组轮换观看，每场 5 分钟，反复多轮
**Demo Mode**: D — Tab1（学习）+ Tab2（推荐）+ 离线渲染好的 day1 视频片段
**Brainstorm log**: `2026-05-02-demo-backup-and-portable-package-brainstorm.md`

---

## 1. 目标与非目标

### 目标

1. **Demo 现场对 LLM/网络故障鲁棒**：OpenAI 直连失败 → OpenRouter → 家里隧道 → 本地 cache → 录屏
2. **可装箱迁移**：通过 U 盘把整个 demo 包从国内台式机搬到法国带过去的 Ubuntu 笔记本，新机器 5 步开箱即跑
3. **多轮幂等**：每场之间 30 秒回到出厂状态，多组观众观感一致
4. **低操作负担**：演示者不需要切模式、不需要做技术决策；紧急切换通过隐蔽 URL query param

### 非目标

- 不重构现有 LLM 调用方（`HabitsDetector` `ProactiveExecutor` 等保持接口）
- 不引入本地大模型推理（笔记本无 GPU）
- 不改视频生成管线（已离线完成）
- 不做生产级监控、SRE 告警

---

## 2. 整体架构

### 2.1 调用链

```
LLMRouter.invoke(prompt, model, temperature)
  │
  ├─ 紧急模式（URL ?cache_only=1 或 env DEMO_FORCE_CACHE=1）
  │     └─ cache.get_or_raise(key)
  │
  ├─ try OpenAIProvider     timeout: connect 5s + read 15s
  ├─ try OpenRouterProvider timeout: connect 5s + read 15s
  ├─ try TunnelProvider     timeout: connect 5s + read 10s
  └─ cache.get_or_raise(key)   ← 三路全断后兜底
```

每次成功调用顺手 `cache.set(key, response)`，让缓存自然增厚。

Embedding 链同构但少一层：`OpenAI → 隧道 → cache`（OpenRouter 不提供 embedding）。

### 2.2 组件清单

| 组件 | 路径 | 职责 |
|---|---|---|
| `LLMRouter` | `panoramix_core/llm_router.py` (新) | 调用链调度，对外保持 `LLMClient.invoke()` 兼容签名 |
| `EmbeddingRouter` | `panoramix_core/embedding_router.py` (新) | embedding 调度 |
| `LLMCache` | `panoramix_core/llm_cache.py` (新) | 读写 `storage/llm_cache.json` 与 `storage/embedding_cache.json` |
| `OpenAIProvider` | `panoramix_core/providers/openai_provider.py` (新) | 现有 OpenAI SDK 调用 + 显式 timeout |
| `OpenRouterProvider` | `panoramix_core/providers/openrouter_provider.py` (新) | OpenRouter base_url + model 映射 |
| `TunnelProvider` | `panoramix_core/providers/tunnel_provider.py` (新) | 走 cloudflared 转发到家里 OpenAI proxy |
| `home_proxy/proxy.py` | repo 顶层 (新) | FastAPI thin proxy + bearer token 校验 |
| Sidebar indicator | `simulator/components/cache_status.py` (新) | 显示 `live ✓` / `CACHE ⚠` |

### 2.3 缓存键 / 值

- key = `sha256(model + "\n" + prompt + "\n" + str(temperature))`
- value = `{"response": str, "cached_at": iso8601, "provider": "openai|openrouter|tunnel|seed"}`
- 文件：`storage/llm_cache.json`（dict 直接序列化），`storage/embedding_cache.json` 同结构（值是 list[float]）

### 2.4 紧急 cache-only 切换

- **入口**：URL query param `?cache_only=1`（演示者地址栏改一下即生效，最隐蔽）；环境变量 `DEMO_FORCE_CACHE=1`（启动时锁定，备用）
- **指示器**：sidebar 底部一行小灰字
  - 主链工作 → `live ✓`
  - 触发降级 / 强制 cache → `CACHE ⚠`
- **行为差异**：
  - **紧急模式**：跳过三个 provider，直接查 cache；miss 即抛错（避免误触发外网）
  - **正常模式**：依次试 OpenAI → OpenRouter → 隧道，三路全断后查 cache 兜底；miss 才抛错
  - 两种模式都有 cache 作为终点，区别在于"是否先尝试外网"

### 2.5 OpenRouter model 映射

`config/openrouter_model_map.yaml`：

```yaml
gpt-4.1: openai/gpt-4-turbo
text-embedding-ada-002: <not_supported>   # embedding 不走 OpenRouter
```

---

## 3. 迁移包结构

### 3.1 U 盘文件夹布局

```
habit-memory-demo-2026-05-XX/
├── README.md                         5 步开箱跑、故障排查
├── README_DEMO_RUNBOOK.md            当天 checklist
├── docs/DEMO_BACKUP_PLAN.md          人话版方案，给 stakeholder 看
├── habit_memory_demo/                项目源码（当前仓库内容）
├── wheels/                           预下载的所有 pip 包（200–500 MB）
├── fixtures/                         frozen demo 数据
│   ├── habit_memory.db
│   ├── chroma_memories/              ChromaDB 目录（不打 tar）
│   ├── llm_cache.json                预热好的 LLM 响应
│   └── embedding_cache.json          预计算 embedding
├── demo_videos/                      离线渲染好的 mp4
├── recordings/                       兜底录屏（用户自录）
│   └── full_demo_<date>.mp4
└── scripts/setup.sh                  一键安装脚本入口
```

`.env` 文件**不进 U 盘**，只放 `.env.example`，到现场手填。

`home_proxy/` 留在国内台式机，不进 U 盘。

### 3.2 5 步开箱

```bash
# 1. 拷贝
cp -r /media/usb/habit-memory-demo-2026-05-XX ~/

# 2. 装环境（离线，从 wheels/ 装）
cd ~/habit-memory-demo-2026-05-XX
bash scripts/setup.sh

# 3. 配 key
cp .env.example .env
nano .env   # 填 OPENAI_API_KEY / OPENROUTER_API_KEY / TUNNEL_BASE_URL / TUNNEL_OPENAI_KEY

# 4. 自检
bash scripts/verify.sh

# 5. 开演
bash scripts/run_demo.sh
```

目标：< 15 分钟。

### 3.3 关键文件细节

**`requirements.txt`**：严格 pin 主依赖版本（streamlit, openai, chromadb, pydantic, numpy, matplotlib, requests, pyjwt, pyyaml, fastapi, uvicorn, httpx）

**`requirements-lock.txt`**：`pip-compile` 输出的全量传递依赖锁定

**`wheels/`**：在 dev 台式机执行 `pip download -r requirements-lock.txt -d wheels/ --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:`

**`.env.example`**：
```bash
# === 主路 LLM ===
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1

# === 备用 1：OpenRouter ===
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# === 备用 2：家里隧道 ===
TUNNEL_BASE_URL=                # 形如 https://xxx.trycloudflare.com/v1
TUNNEL_OPENAI_KEY=              # 跟主 key 隔离

# === 模型 ===
HABIT_DETECTION_LLM_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-ada-002

# === 紧急模式（隐蔽，平时留空）===
DEMO_FORCE_CACHE=

# === LLM Cache（demo 期间只读防污染）===
LLM_CACHE_READ_ONLY=1

# === DBSCAN（一般不动）===
HYBRID_ALPHA=0.8
DBSCAN_EPS=0.10
DBSCAN_MIN_SAMPLES=3
```

---

## 4. 脚本规约

| 脚本 | 行为 |
|---|---|
| `scripts/setup.sh` | 严格校验 Python 3.11 → 建 venv → 离线 pip install from wheels/ → 解 fixtures 到 storage/ |
| `scripts/verify.sh` | 三路 key 通不通（curl /v1/models）+ 视频 mp4 在不在 + cache 文件大小非零 |
| `scripts/run_demo.sh` | `lsof -i :8501` 判端口 + `streamlit run simulator/app.py` + `xset s off` 防息屏 + 自动开浏览器 |
| `scripts/reset_demo.sh` | 杀 streamlit → 删 storage/{habit_memory.db, habit_memory.db-wal, habit_memory.db-shm, memories/} → 从 fixtures/ 拷回 → 重启 streamlit |
| `home_proxy/start.sh` | 启动 FastAPI proxy（监听本地端口）+ cloudflared tunnel，输出 https URL |

`scripts/reset_demo.sh` 必须**幂等**且**端到端**，有自动化测试 `tests/test_demo_reset_idempotent.py` 覆盖。

---

## 5. Demo Day Runbook 关键节点

详见 `README_DEMO_RUNBOOK.md`，本节列高优先动作：

- **T-24h**：verify.sh + 完整 dry-run + **录屏复核**（如未录屏当晚补录）
- **T-1h**：接电源 + 过 captive portal + 隧道连通性测试 + 浏览器双标签预备
- **每场前**：`reset_demo.sh` → 浏览器刷新
- **每场中**：sidebar `live ✓` / `CACHE ⚠` 监控；任一调用 > 8s 沉默就改 URL 加 `?cache_only=1`
- **极端情况**：本地 `recordings/full_demo_<date>.mp4` + 演示者解说

> ⚠️ **录屏文件 `recordings/full_demo_<date>.mp4` 是最后保险**，T-24h checklist 第 6 项强制核查，U 盘 + 笔记本双份存放。

---

## 6. Risk Register（必修项 🔴）

| # | 风险 | 缓解 |
|---|---|---|
| 1 | OpenAI 直连超时 / 慢 | 三层降级链 + 显式 timeout |
| 2 | OpenAI key 被封 / quota 用尽 | OpenRouter 自动接管；T-24h 查 dashboard 用量 |
| 6 | 会场 Wi-Fi captive portal | T-1h checklist 强制过 portal |
| 9 | 投影仪接口不匹配 | 带 USB-C ↔ HDMI/VGA/DP 三种转接头 |
| 11 | 法国电源插头 | 万能转换头 ×2 |
| 12 | 笔记本电池没电 | 电源选项设"接电时永不休眠" |
| 15 | Python 版本不兼容 wheels | setup.sh 严格校验 3.11；wheels 按 manylinux2014 + py3.11 下载 |
| 17 | ChromaDB 版本不兼容 | 严格 pin；fixtures 与代码同版本 |
| 21 | 多场之间 reset 失败 / state 串场 | reset_demo.sh 端到端 + 自动化测试 |
| 30 | U 盘里有 `.env` 含真实 key | U 盘**只放** `.env.example`，现场手填 |
| 33 | U 盘丢失 | 双 U 盘分开存放 + 笔记本本地副本 |

完整 36 条见 brainstorm 日志 Section 4。

---

## 7. 时间表

| 周 | 范围 | 内容 |
|---|---|---|
| Week 1（5/3 – 5/9） | 工程化主体 | LLMRouter + 三层 fallback + 显式 timeout + cache + 隐蔽开关 + indicator + home_proxy |
| Week 2（5/10 – 5/16） | 装包 + 演练 | requirements/wheels/scripts + fixtures + 文档 + **借机干净 dry-run** + reset 幂等性测试 + 视频集成 + 录屏复核 |
| Week 3（5/17 – 5/23） | 法国适配 | 酒店 dry-run + 现场踩点 + 启动家里 cloudflared + 多场演示 |

**P0 freeze 5/9，P0+P1+P2 freeze 5/16。Week 3 不做新工程。**

最坏情况（任何一个 P0 在 5/16 还没完成）：**全 cache 模式 + 录屏兜底**就足够支撑 demo，不要硬上半成品 backup。

---

## 8. 测试策略

| 测试 | 覆盖 |
|---|---|
| `tests/test_llm_router.py` | 三种 provider 模拟失败，验证降级路径正确 |
| `tests/test_llm_cache.py` | 命中、miss、写入、紧急模式 |
| `tests/test_embedding_router.py` | OpenAI 失败 → 隧道 → cache |
| `tests/test_demo_reset_idempotent.py` | 多次 reset 回到完全一致状态 |
| `tests/test_setup_offline.py` | （可选）docker 内执行 setup.sh 全离线安装成功 |
| 手动 dry-run | 借机干净 Ubuntu 22.04，5 步开箱跑通 5 分钟 demo |

---

## 9. 交付清单（Week 2 末尾）

- [ ] U 盘 ×2 含完整迁移包
- [ ] 笔记本本地有同一份副本
- [ ] 录屏 mp4 在 `recordings/` 就位
- [ ] `verify.sh` 三路全通过
- [ ] 借机 dry-run 通过
- [ ] reset 幂等性测试通过
- [ ] 国内台式机 cloudflared 启动脚本就绪 + 24h 开机
- [ ] `.env.example` 不含真实 key
- [ ] 三种插头转换器、三种投屏接口都备齐
