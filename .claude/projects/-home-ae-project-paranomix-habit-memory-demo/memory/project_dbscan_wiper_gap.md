---
name: DBSCAN wiper condition gap
description: 当前 DBSCAN 无法建模「雨刮器≠max时才开窗」的排除条件，StructuredContext 缺少 wiper_state 维度
type: project
---

目标 habit: 接近收费口/停车场入口 + 车速<5kph + 雨刮器没有开到最大 → 开窗

当前 StructuredContext 5 维度 vs 3 个条件的覆盖情况:
- geofence=office_gate_01 → **能** (已在 ctx)
- vehicle_state=crawling (<5kph) → **能** (已在 ctx)
- wiper_state≠max → **不能** (不在 ctx 中)

当前结果"看起来正确"是因为 mockup 数据本身已做排除 — Rainy 天的 unified 文件里没有 window_open_request 事件，所以管线里不会产生大雨天开窗的 Fact。DBSCAN 聚到的全是 wiper=off 时的正例，但模型不知道为什么对。

若未来数据出现大雨天手动开窗（异常操作），该 Fact 的 ctx 与正常开窗完全相同，DBSCAN 无法区分。

**Why:** StructuredContext 只有 time_bucket/hour/weekday/vehicle_state/geofence，hybrid distance 不感知 wiper。
**How to apply:** 若需系统原生识别「大雨排除」���需将 wiper_state 加入 StructuredContext 并参与 context_distance 计算。加维度时须注意权重设置，避免 wiper=low（小雨）也被排除。
