#!/usr/bin/env python3
"""15W と MAXN_SUPER の測定結果を突き合わせて集計する.

入力は fgpu0 で走らせた TRT 10.3 の測定 JSON。両モードとも同じ TensorRT・同じ L4T・
同じプロトコルで測っているので、**電力モードの差だけを純粋に切り出せる**のが
この集計の要点である (論文の既存値は JetPack 5.1.2 / TRT 8.5.2 なので混ぜてはいけない)。

  CNN 56 構成 : results_trt10_15w/      results_trt10_maxn/       1 構成 = 1 エンジン
  ViT-L       : results_trt10_vitl_15w/ results_trt10_vitl_maxn/  1 構成 = **複数パートの連鎖**

出力: results/orin/powermode_trt10.json

⭐ ViT-L はパートの合算が要る
  ViT-L はエンジンを分割しないとエッジに載らないので、1 構成の推論時間は
  各パートの中央値の**総和**である (規約は collect_vitl_edge.py の CHAIN / sweep_rec に従う)。
  エネルギーも同様に **Sum(P_i x t_i)** で積む。「パート平均電力 x 合計時間」ではない
  (パートごとに電力が違うため一致しない)。

⚠️ 欠損パートを 0 として足さない
  collect_vitl_edge.py の sweep_rec は `if ms: tot += ms` なので、パートが 1 つ欠けると
  **黙って短い合計**が出る。ここでは欠損を missing_parts に明示し complete=false にして、
  不完全な連鎖を比較対象から外す。

⚠️ エネルギーの解釈に注意
  VDD_IN はボード全体の入力電力なので、ここで出す mJ は
  「ベンチマーク実行中のボード平均電力 x 推論時間」であって推論だけの
  正味エネルギーではない。**results_trt10_idle.json があれば自動で読み込み**、
  アイドル分を差し引いた net_energy_mJ (推論による増分) も出す。

usage: python3 train/collect_powermode.py [--strict]
       --strict を付けると想定構成数がそろっていないモードがあるときに終了コード 1 を返す
"""
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PM_ROOT は検証用の差し替え口 (本番では指定しない)。
# 測定が片方しか終わっていない段階で「対になったとき」の経路を試すのに使う
R = os.environ.get("PM_ROOT") or os.path.join(HERE, "results", "orin")
OUT = os.path.join(R, "powermode_trt10.json")

# ディレクトリ名 -> JSON 中の power_mode 表記
MODES = {"15W": "results_trt10_15w", "MAXN_SUPER": "results_trt10_maxn"}
MODEL_ORDER = ["mnv4", "effb0", "resnet50", "vit_small"]
RESOLUTIONS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
N_EXPECTED = len(MODEL_ORDER) * len(RESOLUTIONS)  # 56 構成

# ---- ViT-L (分割チェーン) ----
VITL_MODES = {"15W": "results_trt10_vitl_15w", "MAXN_SUPER": "results_trt10_vitl_maxn"}
# 配備構成の段の並び。collect_vitl_edge.py の CHAIN と同じ
#   DINOv2-L は p3 を p3s0/p3s1 に割って **p3s1 だけ FP32** にしてある (FP16 で飽和するため)
#   DINOv3-L は 4 段すべて FP32
VITL_CHAIN = {"dinov2_l": ["p0", "p1", "p2", "p3s0", "p3s1"],
              "dinov3_l": ["p0", "p1", "p2", "p3"]}
VITL_CTRL_RES = [16, 32, 64, 112, 128, 224]
# ⭐ 全段 FP32 対照は **4 段** (p3 を割らない)。旧 engines_split/ には 5 段の `f32_p3s0/p3s1`
#    (計 29) と 4 段の `f32s4_p{0..3}` (計 24) の 2 系統があり、
#    TRT10 の再ビルド (build_trt10_vitl.sh:135-141) が作るのは **4 段の方**である。
#    旧 collect_vitl_edge.py の _chain() は res!=32 で 5 段を要求するので、そちらを
#    新データに当てると p3 段が丸ごと欠落した過小な合計になる。ここでは 4 段で扱う。
VITL_CTRL_CHAIN = ["p0", "p1", "p2", "p3"]
# 単発エンジン (連鎖ではない参照値)。⭐ *_single は「TRT10 なら分割せずに載るか」の検証用
VITL_SINGLE_TAGS = ["vit_small_r112_fp32", "vit_small_r224_fp32",
                    "dinov2_l_r32_single", "dinov3_l_r32_single"]
VITL_N_DEPLOY = len(RESOLUTIONS) * (len(VITL_CHAIN["dinov2_l"])
                                    + len(VITL_CHAIN["dinov3_l"]))   # 126 パート

# アイドル電力 (mW)。results_trt10_idle.json があれば読み込む (無ければ空のまま)
IDLE_MW = {}


def load_idle():
    """アイドル電力を results_trt10_idle.json から読む (idle_power.sh の出力)

    ⚠️ 以前はここが手書きの空 dict だったため、測っても net_energy_mJ が出なかった。
    """
    path = os.path.join(R, "results_trt10_idle.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for mode, rails in (d.get("modes") or {}).items():
        v = (rails or {}).get("VDD_IN", {}).get("mean_mW")
        if v is not None:
            out[mode] = v
    return out


def load_mode(dirname):
    """1 モード分の JSON を読み、tag をキーにした dict で返す"""
    out = {}
    for path in sorted(glob.glob(os.path.join(R, dirname, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        out[d["tag"]] = d
    return out


def metrics(d):
    """1 構成の測定 JSON から論文で使う指標を取り出す"""
    t_ms = d["median_ms"]
    p_mw = d["power"]["VDD_IN"]["mean_mW"] if "power" in d else None
    m = {
        "median_ms": round(t_ms, 5),
        "mean_ms": round(d["mean_ms"], 5),
        "sd_ms": round(d["sd_ms"], 5),
        "cv_pct": d["cv_pct"],
        "n_ok": d["n_ok"],
        "engine_MB": d.get("engine_MB"),
    }
    if p_mw is not None:
        # mW x ms = uJ なので /1000 で mJ
        m["vdd_in_mW"] = p_mw
        m["vdd_cpu_gpu_cv_mW"] = d["power"]["VDD_CPU_GPU_CV"]["mean_mW"]
        m["vdd_soc_mW"] = d["power"]["VDD_SOC"]["mean_mW"]
        m["energy_mJ"] = round(p_mw * t_ms / 1000.0, 4)
        # 1 W あたりのスループット。1e6 / (ms * mW)
        m["fps_per_W"] = round(1e6 / (t_ms * p_mw), 2)
    return m


def net_energy(m, mode):
    """アイドル電力が分かっていれば正味エネルギーも出す"""
    idle = IDLE_MW.get(mode)
    if idle is None or "vdd_in_mW" not in m:
        return None
    return round(max(m["vdd_in_mW"] - idle, 0.0) * m["median_ms"] / 1000.0, 4)


def chain_metrics(byt, tags, mode):
    """分割チェーン 1 本を合算する。tags は段の順に並んだタグ列

    ⭐ 推論時間は各段の中央値の**総和**、エネルギーは **Sum(P_i x t_i)**。
       ボード電力は段ごとに違うので、「平均電力 x 合計時間」では一致しない。
       ここで出す vdd_in_mW は時間で重み付けした平均 (= energy / t_total) である。
    ⚠️ 欠損段は 0 として足さない。missing_parts に残し complete=false にする。
    """
    parts, missing = {}, []
    t_total, e_uj, e_net_uj = 0.0, 0.0, 0.0
    n_ok_min, cv_max, mb_total = None, None, 0.0
    has_power, has_net = True, True
    idle = IDLE_MW.get(mode)

    for tag in tags:
        d = byt.get(tag)
        if d is None or not d.get("n_ok"):
            missing.append(tag)
            continue
        t = d["median_ms"]
        parts[tag] = round(t, 5)
        t_total += t
        mb_total += d.get("engine_MB") or 0.0
        n_ok_min = d["n_ok"] if n_ok_min is None else min(n_ok_min, d["n_ok"])
        cv_max = d["cv_pct"] if cv_max is None else max(cv_max, d["cv_pct"])
        p = (d.get("power") or {}).get("VDD_IN", {}).get("mean_mW")
        if p is None:
            has_power = False
        else:
            e_uj += p * t
            if idle is None:
                has_net = False
            else:
                e_net_uj += max(p - idle, 0.0) * t

    if missing or not parts:
        return {"complete": False, "missing_parts": missing,
                "n_parts_found": len(parts), "n_parts_expected": len(tags)}

    m = {
        "complete": True,
        "n_parts": len(tags),
        "parts_ms": parts,
        "latency_ms": round(t_total, 5),
        "n_ok_min": n_ok_min,
        "cv_pct_max": cv_max,
        "engine_MB_total": round(mb_total, 2),
    }
    if has_power:
        m["energy_mJ"] = round(e_uj / 1000.0, 4)
        # 時間で重み付けした平均ボード電力。energy / t_total と定義が一致する
        m["vdd_in_mW"] = round(e_uj / t_total, 1)
        m["fps_per_W"] = round(1e6 / (t_total * (e_uj / t_total)), 2)
        if has_net and idle is not None:
            m["net_energy_mJ"] = round(e_net_uj / 1000.0, 4)
    return m


def vitl_chains():
    """測るべき ViT-L チェーンの一覧を (キー, モデル, 解像度, 種別, タグ列) で返す"""
    out = []
    for res in RESOLUTIONS:
        out.append(("dinov2_l_r%d" % res, "dinov2_l", res, "deploy",
                    ["dinov2_l_r%d_%s" % (res, p) for p in VITL_CHAIN["dinov2_l"]]))
        out.append(("dinov3_l_r%d" % res, "dinov3_l", res, "deploy",
                    ["dinov3_l_r%d_%s" % (res, p) for p in VITL_CHAIN["dinov3_l"]]))
    for res in VITL_CTRL_RES:
        out.append(("dinov2_l_r%d_f32" % res, "dinov2_l_f32", res, "ctrl",
                    ["dinov2_l_r%d_f32_%s" % (res, p) for p in VITL_CTRL_CHAIN]))
    for tag in VITL_SINGLE_TAGS:
        out.append((tag, tag.rsplit("_r", 1)[0], None, "single", [tag]))
    return out


def spread(values):
    """min/median/max をまとめて返す (論文で「x--y 倍」と書くため)"""
    if not values:
        return None
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def vitl_section():
    """ViT-L 分割チェーンの集計。測定 JSON が 1 件も無ければ None を返す"""
    data = {mode: load_mode(d) for mode, d in VITL_MODES.items()}
    if not any(data.values()):
        return None

    specs = vitl_chains()
    by_chain, paired, incomplete = {}, [], []
    for key, model, res, kind, tags in specs:
        entry = {"model": model, "kind": kind}
        if res is not None:
            entry["res"] = res
        for mode in VITL_MODES:
            if not data[mode]:
                continue
            m = chain_metrics(data[mode], tags, mode)
            entry[mode] = m
            if not m["complete"] and m.get("n_parts_found"):
                # 一部だけ測れている = 測定中か取りこぼし。どちらか判別できないので記録する
                incomplete.append("%s/%s (%d/%d)" % (mode, key, m["n_parts_found"],
                                                     m["n_parts_expected"]))
        a, b = entry.get("15W"), entry.get("MAXN_SUPER")
        if a and b and a.get("complete") and b.get("complete"):
            entry["speedup_maxn_over_15w"] = round(a["latency_ms"] / b["latency_ms"], 4)
            if "vdd_in_mW" in a and "vdd_in_mW" in b:
                entry["power_ratio"] = round(b["vdd_in_mW"] / a["vdd_in_mW"], 4)
                entry["energy_ratio"] = round(b["energy_mJ"] / a["energy_mJ"], 4)
            paired.append(entry)
        if len(entry) > 2:
            by_chain[key] = entry

    # 配備構成だけをまとめる (対照・単発は電力モード比較の主対象ではない)
    dep = [e for e in paired if e["kind"] == "deploy"]
    by_model = {}
    for model in ("dinov2_l", "dinov3_l"):
        rows = [e for e in dep if e["model"] == model]
        if not rows:
            continue
        by_model[model] = {
            "n_paired": len(rows),
            "speedup_maxn_over_15w": spread([e["speedup_maxn_over_15w"] for e in rows]),
            "power_ratio": spread([e["power_ratio"] for e in rows if "power_ratio" in e]),
            "energy_ratio": spread([e["energy_ratio"] for e in rows if "energy_ratio" in e]),
        }

    coverage = {}
    for mode, byt in data.items():
        n_dep = sum(1 for k in byt
                    if not k.endswith("_single") and "_f32_" not in k
                    and not k.startswith("vit_small"))
        coverage[mode] = {"n_parts": len(byt), "n_deploy_parts": n_dep,
                          "of_deploy": VITL_N_DEPLOY,
                          "complete": n_dep == VITL_N_DEPLOY}

    overall = {}
    if dep:
        overall = {
            "n_paired": len(dep),
            "speedup_maxn_over_15w": spread([e["speedup_maxn_over_15w"] for e in dep]),
            "power_ratio": spread([e["power_ratio"] for e in dep if "power_ratio" in e]),
            "energy_ratio": spread([e["energy_ratio"] for e in dep if "energy_ratio" in e]),
        }

    return {
        "note": ("ViT-L は分割しないとエッジに載らないため、1 構成の推論時間は各段の"
                 "中央値の総和、エネルギーは Sum(P_i x t_i) で積んである。"
                 "vdd_in_mW は時間で重み付けした平均ボード電力 (= energy / latency)"),
        "chain_def": {"dinov2_l": VITL_CHAIN["dinov2_l"], "dinov3_l": VITL_CHAIN["dinov3_l"],
                      "ctrl_fp32": VITL_CTRL_CHAIN},
        "ctrl_note": ("全段 FP32 対照は 4 段 (p3 を割らない)。旧 engines_split/ には 5 段の "
                      "f32_p3s0/p3s1 系 (計 29) と 4 段の f32s4 系 (計 24) があり、"
                      "TRT10 の再ビルドが作るのは 4 段の方である。"
                      "旧 collect_vitl_edge.py の _chain() は res!=32 で 5 段を要求するため、"
                      "そのまま新データに当てると p3 段が欠落した過小な合計になる"),
        "coverage": coverage,
        "incomplete_chains": incomplete,
        "overall_deploy": overall,
        "by_model": by_model,
        "by_chain": by_chain,
    }


def main():
    strict = "--strict" in sys.argv
    global IDLE_MW
    IDLE_MW = load_idle()
    data = {mode: load_mode(d) for mode, d in MODES.items()}
    coverage = {
        mode: {"n": len(v), "of": N_EXPECTED, "complete": len(v) == N_EXPECTED}
        for mode, v in data.items()
    }
    for mode, c in coverage.items():
        state = "完了" if c["complete"] else "測定中"
        print(f"[{mode:11s}] {c['n']:2d}/{c['of']} 構成  ({state})")

    # 環境情報は最初に見つかった 1 件から採る (全件同一であることも検査する)
    env, env_mismatch = {}, []
    for mode, byt in data.items():
        for tag, d in byt.items():
            e = {"trt": d.get("trt"), "l4t": d.get("l4t"), "reps": d.get("reps"),
                 "protocol": d.get("protocol")}
            if not env:
                env = e
            elif e != env:
                env_mismatch.append(f"{mode}/{tag}")

    by_config, paired = {}, []
    for model in MODEL_ORDER:
        for res in RESOLUTIONS:
            tag = f"{model}_r{res}"
            entry = {"model": model, "res": res}
            for mode in MODES:
                if tag in data[mode]:
                    m = metrics(data[mode][tag])
                    ne = net_energy(m, mode)
                    if ne is not None:
                        m["net_energy_mJ"] = ne
                    entry[mode] = m
            a, b = entry.get("15W"), entry.get("MAXN_SUPER")
            if a and b:
                entry["speedup_maxn_over_15w"] = round(a["median_ms"] / b["median_ms"], 4)
                if "vdd_in_mW" in a and "vdd_in_mW" in b:
                    entry["power_ratio"] = round(b["vdd_in_mW"] / a["vdd_in_mW"], 4)
                    entry["energy_ratio"] = round(b["energy_mJ"] / a["energy_mJ"], 4)
                paired.append(entry)
            if len(entry) > 2:
                by_config[tag] = entry

    # モデル別の要約。論文では「モデル別に x--y 倍」と書くのでレンジが要る
    by_model = {}
    for model in MODEL_ORDER:
        rows = [e for e in paired if e["model"] == model]
        if not rows:
            continue
        by_model[model] = {
            "speedup_maxn_over_15w": spread([e["speedup_maxn_over_15w"] for e in rows]),
            "power_ratio": spread([e["power_ratio"] for e in rows if "power_ratio" in e]),
            "energy_ratio": spread([e["energy_ratio"] for e in rows if "energy_ratio" in e]),
        }

    overall = {}
    if paired:
        overall = {
            "n_paired": len(paired),
            "speedup_maxn_over_15w": spread([e["speedup_maxn_over_15w"] for e in paired]),
            "power_ratio": spread([e["power_ratio"] for e in paired if "power_ratio" in e]),
            "energy_ratio": spread([e["energy_ratio"] for e in paired if "energy_ratio" in e]),
        }

    vitl = vitl_section()

    out = {
        "note": ("Jetson Orin Nano の 15W と MAXN_SUPER を同一条件 (JetPack 6.2 / "
                 "L4T R36.4.3 / TensorRT 10.3) で比較した集計。論文の既存値は "
                 "JetPack 5.1.2 / TRT 8.5.2 なので混ぜてはいけない"),
        "energy_caveat": ("energy_mJ は VDD_IN (ボード入力電力) の実行中平均に推論時間を"
                          "掛けたもので、推論だけの正味エネルギーではない。"
                          "IDLE_MW を埋めると net_energy_mJ も出る"),
        "env": env,
        "env_mismatch": env_mismatch,
        "idle_mW": IDLE_MW or None,
        "coverage": coverage,
        "overall": overall,
        "by_model": by_model,
        "by_config": by_config,
    }
    # ViT-L はデータがあるときだけ足す (無いときは CNN 専用だった頃と同じ出力になる)
    if vitl is not None:
        out["vitl"] = vitl
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[saved] {os.path.relpath(OUT, HERE)}  (対になった構成 {len(paired)} 件)")

    if env_mismatch:
        print(f"[警告] 環境情報が揃っていない構成が {len(env_mismatch)} 件: "
              f"{env_mismatch[:3]}")
    if overall:
        s = overall["speedup_maxn_over_15w"]
        print(f"  MAXN/15W 高速化: {s['min']}--{s['max']} 倍 (中央値 {s['median']})")
        if overall.get("energy_ratio"):
            e = overall["energy_ratio"]
            print(f"  エネルギー比    : {e['min']}--{e['max']} 倍 (中央値 {e['median']})")
    else:
        print("  ※ まだ両モードが揃った構成が無いので比は出していない")

    if IDLE_MW:
        print("  アイドル電力    : "
              + " / ".join(f"{m} {v} mW" for m, v in IDLE_MW.items())
              + "  -> net_energy_mJ を算出")
    else:
        print("  ※ results_trt10_idle.json が無いので net_energy_mJ は出していない")

    if vitl is not None:
        print("[ViT-L]")
        for mode, c in vitl["coverage"].items():
            state = "完了" if c["complete"] else "測定中"
            print(f"  [{mode:11s}] 配備 {c['n_deploy_parts']:3d}/{c['of_deploy']} パート"
                  f" (全 {c['n_parts']} 件) ({state})")
        vo = vitl["overall_deploy"]
        if vo:
            s = vo["speedup_maxn_over_15w"]
            print(f"  配備チェーン n={vo['n_paired']}  "
                  f"MAXN/15W 高速化 {s['min']}--{s['max']} 倍 (中央値 {s['median']})")
            if vo.get("energy_ratio"):
                e = vo["energy_ratio"]
                print(f"  エネルギー比 {e['min']}--{e['max']} 倍 (中央値 {e['median']})")
        if vitl["incomplete_chains"]:
            print(f"  ⚠️ 段が欠けている連鎖が {len(vitl['incomplete_chains'])} 件 "
                  f"(比較から除外): {vitl['incomplete_chains'][:3]}")

    if strict:
        ng = [m for m, c in coverage.items() if not c["complete"]]
        if vitl is not None:
            ng += [f"vitl/{m}" for m, c in vitl["coverage"].items() if not c["complete"]]
        if ng:
            print(f"[NG] --strict 指定だが想定構成数に足りないモードがある: {ng}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
