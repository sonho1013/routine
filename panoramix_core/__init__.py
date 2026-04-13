"""
panoramix_core — 适配自 Panoramix Memory Service

对齐 Panoramix 架构，适配车载信号习惯记忆场景：
- Fact 模型增加 structured context (time/location/vehicle_state)
- HabitsDetector 使用 hybrid distance (text embedding + structured context)
- Embedder 使用 OpenAI ada-002（通过环境变量配置）
"""
