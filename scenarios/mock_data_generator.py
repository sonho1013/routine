"""
Mock 车辆信号数据生成器
模拟真实用户 Mary 的 5 天驾驶行为，包含自然的随机偏差和噪声。

设计原则：
- 人不是机器人：同一习惯每天会有微小变化（温度±1°C、时间±20min、偶尔换歌）
- 包含"非习惯噪声"：随机的操作不应该形成聚类
- 场景之间有清晰的语义边界，但数值可能重叠（如空调温度）
"""
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any


random.seed(42)  # 可复现


# ──────────────────────────────────────────────
# 地理围栏预设
# ──────────────────────────────────────────────
# Shanghai coordinates (对齐 unified 数据的 location_profile)
GEOFENCES = {
    "home":           {"lat": 31.2304, "lon": 121.4737, "radius_m": 200},
    "work":           {"lat": 31.2396, "lon": 121.4997, "radius_m": 100},
    "office_gate_01": {"lat": 31.2388, "lon": 121.4975, "radius_m": 100},
    "park":           {"lat": 31.2270, "lon": 121.4610, "radius_m": 200},
    "mall":           {"lat": 31.2350, "lon": 121.4900, "radius_m": 150},
}


def _jitter(base: float, pct: float = 0.001) -> float:
    """给 GPS 坐标加微小抖动"""
    return base + random.uniform(-base * pct, base * pct)


def _time_jitter(base_hour: int, base_min: int, jitter_min: int = 20) -> tuple:
    """给时间加随机偏移，模拟人类不精确的作息"""
    total_min = base_hour * 60 + base_min + random.randint(-jitter_min, jitter_min)
    return total_min // 60, total_min % 60


# ──────────────────────────────────────────────
# 场景1: 早晨通勤 — Mary 的习惯
# ──────────────────────────────────────────────
# Mary 的基线习惯:
#   - 工作日 7:30-8:30 出发
#   - 空调 22°C（冷天会调高到 23-24）
#   - 座椅加热 Level 2
#   - 导航去公司（最快路线）
#   - 播客 "Tech Daily" 或 "Morning Brew"，偶尔换成音乐
#   - Eco 模式

def generate_morning_commute(day_offset: int, base_date: datetime) -> Dict:
    """生成单天早晨通勤信号序列"""
    date = base_date + timedelta(days=day_offset)
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday = weekday_names[date.weekday()]

    h, m = _time_jitter(8, 0, jitter_min=25)
    start_time = date.replace(hour=h, minute=m, second=0)

    # 天气影响行为
    weather = random.choice(["sunny", "cloudy", "rainy", "cloudy", "sunny"])
    outside_temp = random.uniform(3, 12) if weather in ["rainy", "cloudy"] else random.uniform(10, 18)

    # Mary 的空调偏好：基线 22°C，冷天调高
    if outside_temp < 5:
        hvac_temp = random.choice([23, 24, 24])  # 很冷→调高
    elif outside_temp < 10:
        hvac_temp = random.choice([22, 23, 22])  # 偏冷→偶尔调高
    else:
        hvac_temp = random.choice([22, 22, 21])  # 舒适→稳定 22

    # 座椅加热：冷天开高档
    seat_heat = 2 if outside_temp < 12 else random.choice([1, 0])

    # 媒体：80% 播客，20% 音乐
    if random.random() < 0.8:
        media_type = "podcast"
        media_content = random.choice(["Tech Daily", "Tech Daily", "Morning Brew", "Tech Daily"])
    else:
        media_type = "music"
        media_content = random.choice(["Jazz Morning Mix", "Lo-fi Focus"])

    media_volume = random.randint(55, 70)

    # 导航
    nav_dest = "work"
    nav_route = "fastest"

    # 驾驶模式：几乎总是 eco
    drive_mode = "eco" if random.random() < 0.9 else "normal"

    # ACC：通勤拥堵路段用 short
    acc = "short" if random.random() < 0.7 else "medium"

    signals = []
    t = start_time

    # 上车启动序列
    signals.append({"t": t.isoformat(), "signal": "engine_status", "value": "on"})
    t += timedelta(seconds=random.randint(5, 15))
    signals.append({"t": t.isoformat(), "signal": "door_status", "value": "closed"})
    t += timedelta(seconds=random.randint(3, 8))
    signals.append({"t": t.isoformat(), "signal": "gear_position", "value": "D"})
    t += timedelta(seconds=random.randint(5, 20))

    # 用户操作序列（顺序有随机性）
    ops = [
        {"signal": "hvac_temp_target", "value": hvac_temp},
        {"signal": "hvac_fan_speed", "value": "auto"},
        {"signal": "seat_heating", "value": seat_heat},
        {"signal": "nav_destination", "value": nav_dest},
        {"signal": "nav_route_pref", "value": nav_route},
        {"signal": "media_source", "value": media_type},
        {"signal": "media_content_id", "value": media_content},
        {"signal": "media_volume", "value": media_volume},
        {"signal": "drive_mode", "value": drive_mode},
        {"signal": "acc_distance", "value": acc},
    ]
    random.shuffle(ops)  # 人操作顺序不固定

    for op in ops:
        t += timedelta(seconds=random.randint(3, 25))
        signals.append({"t": t.isoformat(), "signal": op["signal"], "value": op["value"]})

    return {
        "scene": "morning_commute",
        "day": day_offset + 1,
        "weekday": weekday,
        "date": date.strftime("%Y-%m-%d"),
        "weather": weather,
        "outside_temp_c": round(outside_temp, 1),
        "context": {
            "time_window": "06:00-10:00",
            "day_type": "weekday",
            "vehicle_state": "engine_start",
            "location": "home_departure",
        },
        "signals": signals,
    }


# ──────────────────────────────────────────────
# 场景2: 到家离车 — Mary 的习惯
# ──────────────────────────────────────────────
# Mary 的基线习惯:
#   - 工作日 18:00-19:30 到家
#   - 关空调、关窗、关闭无钥匙进入
#   - 停好车后音频淡出
#   - 偶尔忘记关窗（噪声）

def generate_home_arrival(day_offset: int, base_date: datetime) -> Dict:
    date = base_date + timedelta(days=day_offset)
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday = weekday_names[date.weekday()]

    h, m = _time_jitter(18, 30, jitter_min=35)
    start_time = date.replace(hour=h, minute=m, second=0)

    signals = []
    t = start_time

    # 接近家：GPS 进入围栏
    home = GEOFENCES["home"]
    signals.append({"t": t.isoformat(), "signal": "gps_latitude", "value": _jitter(home["lat"])})
    signals.append({"t": t.isoformat(), "signal": "gps_longitude", "value": _jitter(home["lon"])})
    signals.append({"t": t.isoformat(), "signal": "vehicle_speed", "value": random.randint(8, 20)})
    t += timedelta(seconds=random.randint(30, 90))

    # 停车
    signals.append({"t": t.isoformat(), "signal": "vehicle_speed", "value": 0})
    signals.append({"t": t.isoformat(), "signal": "gear_position", "value": "P"})
    t += timedelta(seconds=random.randint(5, 15))

    # 用户操作
    signals.append({"t": t.isoformat(), "signal": "media_source", "value": "off"})
    signals.append({"t": t.isoformat(), "signal": "media_volume", "value": 0})
    t += timedelta(seconds=random.randint(3, 10))

    signals.append({"t": t.isoformat(), "signal": "hvac_power", "value": "off"})
    t += timedelta(seconds=random.randint(5, 15))

    # 关窗（90% 概率关，10% 忘了 — 噪声）
    if random.random() < 0.9:
        signals.append({"t": t.isoformat(), "signal": "window_position", "value": 0})
    t += timedelta(seconds=random.randint(3, 10))

    signals.append({"t": t.isoformat(), "signal": "keyless_entry", "value": "disabled"})
    t += timedelta(seconds=random.randint(5, 20))

    # 熄火开门
    signals.append({"t": t.isoformat(), "signal": "engine_status", "value": "off"})
    t += timedelta(seconds=random.randint(5, 15))
    signals.append({"t": t.isoformat(), "signal": "door_status", "value": "open"})
    t += timedelta(seconds=random.randint(10, 30))
    signals.append({"t": t.isoformat(), "signal": "door_status", "value": "locked"})

    return {
        "scene": "arriving_home",
        "day": day_offset + 1,
        "weekday": weekday,
        "date": date.strftime("%Y-%m-%d"),
        "context": {
            "time_window": "17:00-21:00",
            "day_type": "weekday",
            "vehicle_state": "parking",
            "location": "home",
        },
        "signals": signals,
    }


# ──────────────────────────────────────────────
# 场景3: 收费站/停车场入口开窗
# ──────────────────────────────────────────────
# Mary 的习惯:
#   - 接近收费站/停车场时低速 → 开驾驶员窗到 ~80%
#   - 大雨时不开（雨刮最大档）
#   - 开窗幅度有波动（70%-85%）

def generate_toll_entry(event_index: int, base_date: datetime) -> Dict:
    date = base_date + timedelta(days=event_index)
    h, m = _time_jitter(random.choice([9, 12, 15, 17]), 0, jitter_min=30)
    start_time = date.replace(hour=h, minute=m, second=0)

    # 随机选择 POI (对齐 GEOFENCES 中的实际 key)
    poi_name = random.choice(["office_gate_01", "mall", "office_gate_01", "office_gate_01"])
    poi = GEOFENCES[poi_name]

    # 天气
    weather = random.choice(["sunny", "cloudy", "light_rain", "sunny", "cloudy"])
    wiper = "off" if weather == "sunny" else ("low" if weather in ["cloudy", "light_rain"] else "high")

    # 车速 < 5
    speed = random.uniform(1.5, 4.8)

    # 开窗幅度：70-85% 波动
    window_pct = random.randint(70, 85)

    signals = []
    t = start_time

    # 接近 POI
    signals.append({"t": t.isoformat(), "signal": "gps_latitude", "value": _jitter(poi["lat"])})
    signals.append({"t": t.isoformat(), "signal": "gps_longitude", "value": _jitter(poi["lon"])})
    signals.append({"t": t.isoformat(), "signal": "vehicle_speed", "value": round(speed, 1)})
    signals.append({"t": t.isoformat(), "signal": "wiper_state", "value": wiper})
    t += timedelta(seconds=random.randint(3, 10))

    # 用户开窗
    signals.append({"t": t.isoformat(), "signal": "window_position", "value": window_pct})

    return {
        "scene": "toll_parking_entry",
        "day": event_index + 1,
        "date": date.strftime("%Y-%m-%d"),
        "poi": poi_name,
        "weather": weather,
        "wiper": wiper,
        "speed_kph": round(speed, 1),
        "context": {
            "vehicle_state": "approaching_poi",
            "location": poi_name.replace("_", " "),
            "speed_below_5kph": True,
            "wiper_not_max": wiper != "max",
        },
        "signals": signals,
    }


# ──────────────────────────────────────────────
# 噪声数据：非习惯的随机操作
# ──────────────────────────────────────────────
def generate_noise_events(base_date: datetime, count: int = 8) -> List[Dict]:
    """
    模拟零散的非习惯操作：
    - 周末随便开车调了下温度
    - 午休时挪车
    - 晚上临时出门买东西
    这些操作不应该形成聚类
    """
    noises = []
    for i in range(count):
        date = base_date + timedelta(days=random.randint(0, 6))
        h = random.choice([10, 11, 13, 14, 15, 20, 21, 22])
        t = date.replace(hour=h, minute=random.randint(0, 59), second=0)

        signals = []
        signals.append({"t": t.isoformat(), "signal": "engine_status", "value": "on"})

        # 随机操作 1-3 个
        possible_ops = [
            {"signal": "hvac_temp_target", "value": random.randint(18, 28)},
            {"signal": "media_source", "value": random.choice(["radio", "music", "off"])},
            {"signal": "media_content_id", "value": random.choice(["FM 98.5", "Random Playlist", "News Radio"])},
            {"signal": "window_position", "value": random.choice([0, 30, 50, 100])},
            {"signal": "drive_mode", "value": random.choice(["normal", "sport", "eco"])},
            {"signal": "nav_destination", "value": random.choice(["supermarket", "gym", "restaurant"])},
            {"signal": "seat_heating", "value": random.choice([0, 1, 3])},
        ]
        selected = random.sample(possible_ops, k=random.randint(1, 3))
        for op in selected:
            t += timedelta(seconds=random.randint(5, 30))
            signals.append({"t": t.isoformat(), **op})

        noises.append({
            "scene": "noise",
            "day": i + 1,
            "date": date.strftime("%Y-%m-%d"),
            "context": {"time_window": "random", "day_type": "any", "note": "non-habitual"},
            "signals": signals,
        })

    return noises


# ──────────────────────────────────────────────
# 生成完整 5 天 mock 数据集
# ──────────────────────────────────────────────
def generate_full_dataset(base_date: datetime = None) -> Dict:
    if base_date is None:
        base_date = datetime(2025, 10, 6)  # 周一开始

    dataset = {
        "user": "Mary",
        "user_id": "U001",
        "base_date": base_date.strftime("%Y-%m-%d"),
        "scenes": {
            "morning_commute": [],
            "arriving_home": [],
            "toll_parking_entry": [],
            "noise": [],
        }
    }

    for day in range(5):
        dataset["scenes"]["morning_commute"].append(
            generate_morning_commute(day, base_date)
        )
        dataset["scenes"]["arriving_home"].append(
            generate_home_arrival(day, base_date)
        )

    for i in range(5):
        dataset["scenes"]["toll_parking_entry"].append(
            generate_toll_entry(i, base_date)
        )

    dataset["scenes"]["noise"] = generate_noise_events(base_date)

    return dataset


if __name__ == "__main__":
    data = generate_full_dataset()
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    print(f"\n=== Dataset Summary ===")
    for scene, events in data["scenes"].items():
        print(f"  {scene}: {len(events)} events")
