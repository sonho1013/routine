"""
eps 参数扫描 + Fact 模板优化验证
基于已保存的 embedding_validation_result.json 中的数据重新加载 embeddings,
或直接用上一轮已有的 embeddings 做多组 eps 扫描。
"""
import os
import sys
import json
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from scenarios.mock_data_generator import generate_full_dataset
from engine.signal_to_fact import scene_signals_to_facts, noise_signals_to_facts


def _fix_socks_proxy():
    for var in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY",
                "all_proxy", "https_proxy", "http_proxy"):
        val = os.environ.get(var, "")
        if val.startswith("socks://"):
            os.environ[var] = val.replace("socks://", "socks5://", 1)


def get_embeddings(texts, model="text-embedding-ada-002"):
    _fix_socks_proxy()
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])


def generate_facts():
    dataset = generate_full_dataset()
    labeled = []
    for scene_type in ["morning_commute", "arriving_home", "toll_parking_entry"]:
        for event in dataset["scenes"][scene_type]:
            facts = scene_signals_to_facts(scene_type, event)
            for f in facts:
                labeled.append({
                    "text": f["text"],
                    "scene": scene_type,
                    "signal": f.get("metadata", {}).get("signal", ""),
                })
    for event in dataset["scenes"]["noise"]:
        for f in noise_signals_to_facts(event):
            labeled.append({"text": f["text"], "scene": "noise", "signal": ""})
    return labeled


# ──── 优化版 Fact 模板 ────
# 思路：在模板中增加更多场景特有的上下文词汇，拉大跨场景 Embedding 距离
ENHANCED_TEMPLATES = {
    "morning_commute": {
        "hvac_temp_target": (
            "during weekday morning commute departure, "
            "set cabin air conditioning temperature to {value} degrees celsius"
        ),
        "seat_heating": (
            "for the morning drive to work, "
            "turned on driver seat heating to level {value}"
        ),
        "nav_destination": (
            "at morning departure, "
            "started turn-by-turn navigation to {value} for daily commute"
        ),
        "nav_route_pref": (
            "for morning commute navigation, selected {value} route preference"
        ),
        "media_podcast": (
            "while commuting to work in the morning, "
            "resumed listening to podcast {content_id}"
        ),
        "media_music": (
            "while commuting to work in the morning, "
            "started playing music {content_id}"
        ),
        "media_volume": (
            "during morning commute drive, "
            "adjusted media playback volume to {value} percent"
        ),
        "drive_mode": (
            "for the morning commute trip, "
            "switched vehicle to {value} driving mode"
        ),
        "acc_distance": (
            "for commute driving, "
            "set adaptive cruise control following distance to {value}"
        ),
    },
    "arriving_home": {
        "hvac_off": (
            "upon arriving at home and parking the vehicle, "
            "shut down the cabin air conditioning system"
        ),
        "media_off": (
            "after reaching home and putting vehicle in park, "
            "stopped all audio media playback"
        ),
        "windows_closed": (
            "after parking at home location, "
            "closed all vehicle windows completely"
        ),
        "keyless_disabled": (
            "after arriving home and exiting the vehicle, "
            "disabled the keyless proximity entry system"
        ),
        "engine_off": (
            "after parking at home, "
            "shut down the vehicle engine"
        ),
    },
    "toll_parking_entry": {
        "window_opened": (
            "when approaching {poi_name} at very low speed below 5 kph, "
            "automatically lowered the driver side window to {value} percent opening"
        ),
    },
}


def generate_enhanced_facts():
    """用优化模板生成 Fact"""
    dataset = generate_full_dataset()
    labeled = []

    for event in dataset["scenes"]["morning_commute"]:
        signals = {s["signal"]: s["value"] for s in event["signals"]}
        tpls = ENHANCED_TEMPLATES["morning_commute"]

        if "hvac_temp_target" in signals:
            labeled.append({"text": tpls["hvac_temp_target"].format(value=signals["hvac_temp_target"]),
                            "scene": "morning_commute", "signal": "hvac_temp_target"})
        if "seat_heating" in signals and signals["seat_heating"] > 0:
            labeled.append({"text": tpls["seat_heating"].format(value=signals["seat_heating"]),
                            "scene": "morning_commute", "signal": "seat_heating"})
        if "nav_destination" in signals:
            labeled.append({"text": tpls["nav_destination"].format(value=signals["nav_destination"]),
                            "scene": "morning_commute", "signal": "nav_destination"})
        if "nav_route_pref" in signals:
            labeled.append({"text": tpls["nav_route_pref"].format(value=signals["nav_route_pref"]),
                            "scene": "morning_commute", "signal": "nav_route_pref"})
        if "media_source" in signals:
            mt = signals["media_source"]
            cid = signals.get("media_content_id", "unknown")
            if mt == "podcast":
                labeled.append({"text": tpls["media_podcast"].format(content_id=cid),
                                "scene": "morning_commute", "signal": "media_source"})
            elif mt == "music":
                labeled.append({"text": tpls["media_music"].format(content_id=cid),
                                "scene": "morning_commute", "signal": "media_source"})
        if "media_volume" in signals:
            labeled.append({"text": tpls["media_volume"].format(value=signals["media_volume"]),
                            "scene": "morning_commute", "signal": "media_volume"})
        if "drive_mode" in signals:
            labeled.append({"text": tpls["drive_mode"].format(value=signals["drive_mode"]),
                            "scene": "morning_commute", "signal": "drive_mode"})
        if "acc_distance" in signals:
            labeled.append({"text": tpls["acc_distance"].format(value=signals["acc_distance"]),
                            "scene": "morning_commute", "signal": "acc_distance"})

    for event in dataset["scenes"]["arriving_home"]:
        signals = {s["signal"]: s["value"] for s in event["signals"]}
        tpls = ENHANCED_TEMPLATES["arriving_home"]
        if signals.get("hvac_power") == "off":
            labeled.append({"text": tpls["hvac_off"], "scene": "arriving_home", "signal": "hvac_power"})
        if signals.get("media_source") == "off" or signals.get("media_volume") == 0:
            labeled.append({"text": tpls["media_off"], "scene": "arriving_home", "signal": "media_off"})
        if "window_position" in signals and signals["window_position"] == 0:
            labeled.append({"text": tpls["windows_closed"], "scene": "arriving_home", "signal": "window_position"})
        if signals.get("keyless_entry") == "disabled":
            labeled.append({"text": tpls["keyless_disabled"], "scene": "arriving_home", "signal": "keyless_entry"})
        if signals.get("engine_status") == "off":
            labeled.append({"text": tpls["engine_off"], "scene": "arriving_home", "signal": "engine_status"})

    for event in dataset["scenes"]["toll_parking_entry"]:
        signals = {s["signal"]: s["value"] for s in event["signals"]}
        tpls = ENHANCED_TEMPLATES["toll_parking_entry"]
        if "window_position" in signals and signals["window_position"] > 0:
            poi_name = event.get("poi", "toll station").replace("_", " ")
            labeled.append({"text": tpls["window_opened"].format(value=signals["window_position"], poi_name=poi_name),
                            "scene": "toll_parking_entry", "signal": "window_position"})

    # 噪声保持原样
    for event in dataset["scenes"]["noise"]:
        for f in noise_signals_to_facts(event):
            labeled.append({"text": f["text"], "scene": "noise", "signal": ""})

    return labeled


def evaluate_dbscan(embeddings, labeled, eps, min_samples=3):
    """运行 DBSCAN 并计算聚类质量"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()

    # 纯度：每个簇中最大场景比例的加权平均
    purity_scores = []
    cluster_info = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        members = [labeled[i] for i in range(len(labels)) if labels[i] == cid]
        scene_counts = Counter(m["scene"] for m in members)
        signal_counts = Counter(m["signal"] for m in members)
        dominant_scene = scene_counts.most_common(1)[0]
        purity = dominant_scene[1] / len(members)
        purity_scores.append(purity)
        cluster_info.append({
            "id": cid,
            "size": len(members),
            "dominant": f"{dominant_scene[0]}:{signal_counts.most_common(1)[0][0]}",
            "purity": purity,
            "scenes": dict(scene_counts),
            "sample": members[0]["text"][:70],
        })

    avg_purity = np.mean(purity_scores) if purity_scores else 0
    return n_clusters, n_noise, avg_purity, cluster_info, labels


def run_sweep():
    print("=" * 70)
    print("  Phase 1: eps 参数扫描 (原始模板)")
    print("=" * 70)

    labeled_orig = generate_facts()
    texts_orig = [f["text"] for f in labeled_orig]

    print(f"\n  Generating embeddings for {len(texts_orig)} original facts...")
    emb_orig = get_embeddings(texts_orig)
    print(f"  Done. dim={emb_orig.shape[1]}")

    eps_candidates = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    print(f"\n  {'eps':<6} {'clusters':<10} {'noise_pts':<10} {'purity':<8}")
    print("  " + "-" * 40)

    best_eps = None
    best_score = -1

    for eps in eps_candidates:
        nc, nn, pur, info, _ = evaluate_dbscan(emb_orig, labeled_orig, eps)
        score = nc * pur - 0.1 * nn  # 简单评分
        marker = ""
        if nc >= 3 and pur >= 0.9:
            if score > best_score:
                best_score = score
                best_eps = eps
                marker = " ← best"
        print(f"  {eps:<6.2f} {nc:<10} {nn:<10} {pur:<8.3f}{marker}")

    # 最优 eps 详情
    if best_eps:
        print(f"\n  最优 eps = {best_eps}")
        nc, nn, pur, info, labels = evaluate_dbscan(emb_orig, labeled_orig, best_eps)
        print(f"  clusters={nc}, noise={nn}, purity={pur:.3f}")
        for ci in info:
            print(f"    Cluster {ci['id']}: size={ci['size']}, "
                  f"dominant={ci['dominant']}, purity={ci['purity']:.2f}")
            print(f"      e.g. \"{ci['sample']}\"")

    # ──── Phase 2: 优化模板 ────
    print("\n" + "=" * 70)
    print("  Phase 2: 优化模板 + eps 扫描")
    print("=" * 70)

    labeled_enh = generate_enhanced_facts()
    texts_enh = [f["text"] for f in labeled_enh]

    print(f"\n  Generating embeddings for {len(texts_enh)} enhanced facts...")
    emb_enh = get_embeddings(texts_enh)
    print(f"  Done. dim={emb_enh.shape[1]}")

    # 先看距离分布
    from itertools import combinations
    norms = np.linalg.norm(emb_enh, axis=1, keepdims=True)
    norm_emb = emb_enh / norms
    dist_mat = 1.0 - norm_emb @ norm_emb.T

    # 同场景同信号距离
    print("\n  同场景同信号距离 (enhanced):")
    scenes = ["morning_commute", "arriving_home", "toll_parking_entry"]
    for scene in scenes:
        by_sig = {}
        for i, f in enumerate(labeled_enh):
            if f["scene"] == scene:
                by_sig.setdefault(f["signal"], []).append(i)
        for sig, idxs in sorted(by_sig.items()):
            if len(idxs) < 2:
                continue
            dists = [dist_mat[i][j] for i, j in combinations(idxs, 2)]
            print(f"    {scene}:{sig:<20} max={max(dists):.4f}")

    # 跨场景最小距离
    print("\n  跨场景最小距离 (enhanced):")
    for s1, s2 in combinations(scenes, 2):
        idx1 = [i for i, f in enumerate(labeled_enh) if f["scene"] == s1]
        idx2 = [i for i, f in enumerate(labeled_enh) if f["scene"] == s2]
        min_d = min(dist_mat[i][j] for i in idx1 for j in idx2)
        print(f"    {s1:<25} ↔ {s2:<25} min={min_d:.4f}")

    # 噪声
    noise_idx = [i for i, f in enumerate(labeled_enh) if f["scene"] == "noise"]
    print("\n  噪声↔场景最小距离 (enhanced):")
    for scene in scenes:
        sidx = [i for i, f in enumerate(labeled_enh) if f["scene"] == scene]
        if noise_idx and sidx:
            min_d = min(dist_mat[i][j] for i in noise_idx for j in sidx)
            print(f"    noise ↔ {scene:<25} min={min_d:.4f}")

    print(f"\n  {'eps':<6} {'clusters':<10} {'noise_pts':<10} {'purity':<8}")
    print("  " + "-" * 40)

    best_eps2 = None
    best_score2 = -1

    for eps in eps_candidates:
        nc, nn, pur, info, _ = evaluate_dbscan(emb_enh, labeled_enh, eps)
        score = nc * pur - 0.1 * nn
        marker = ""
        if nc >= 3 and pur >= 0.9:
            if score > best_score2:
                best_score2 = score
                best_eps2 = eps
                marker = " ← best"
        print(f"  {eps:<6.2f} {nc:<10} {nn:<10} {pur:<8.3f}{marker}")

    if best_eps2:
        print(f"\n  最优 eps = {best_eps2}")
        nc, nn, pur, info, labels = evaluate_dbscan(emb_enh, labeled_enh, best_eps2)
        print(f"  clusters={nc}, noise={nn}, purity={pur:.3f}")
        for ci in info:
            print(f"    Cluster {ci['id']}: size={ci['size']}, "
                  f"dominant={ci['dominant']}, purity={ci['purity']:.2f}")
            print(f"      e.g. \"{ci['sample']}\"")
    else:
        # 找最好的结果展示
        print("\n  未找到满足 clusters>=3 && purity>=0.9 的 eps")
        for eps in eps_candidates:
            nc, nn, pur, info, labels = evaluate_dbscan(emb_enh, labeled_enh, eps)
            if nc >= 3:
                print(f"\n  eps={eps} 的聚类详情:")
                for ci in info:
                    print(f"    Cluster {ci['id']}: size={ci['size']}, "
                          f"dominant={ci['dominant']}, purity={ci['purity']:.2f}")
                    print(f"      e.g. \"{ci['sample']}\"")
                break

    print("\n" + "=" * 70)
    print("  完成。对比原始模板 vs 优化模板的效果。")
    print("=" * 70)


if __name__ == "__main__":
    run_sweep()
