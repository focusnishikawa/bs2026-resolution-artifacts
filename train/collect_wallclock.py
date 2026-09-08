#!/usr/bin/env python3
"""連続実行 wall-clock (査読指摘 A-6b) を合算時間モデルと突き合わせる.

論文の判別時間は「各パートの trtexec GPU 時間の**和**」である。ViT-L は 4-5 段に
分割してあるので、和にはホスト側の呼び出し・同期・段間の受け渡しが入らない。
**10 ms 予算で選ばれる DINOv2-L N=64 は合算 9.056 ms** で余裕が小さいため、
合算モデルが実行時間として通用するかを実機の連続実行で確かめる。

入力: results/wallclock/<label>_rep<N>.txt (1 行 1 枚の ms)
出力: results/wallclock_trt10.json

usage: python3 train/collect_wallclock.py
"""
import json
import os
import re

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(R, "results", "wallclock")
OUT = os.path.join(R, "results", "wallclock_trt10.json")
TABLES = os.path.join(R, "results", "final_tables_trt10_maxn.json")

# 合算時間モデル側の出所 (どこから引くかを明示する)
SUM_OF = {
    "resnet50_r112": ("orin_latency", "resnet50", "112"),
    "resnet50_r224": ("orin_latency", "resnet50", "224"),
    "dinov2_l_r64":  ("sweep", "dinov2_l", "64"),
    "dinov2_l_r224": ("sweep", "dinov2_l", "224"),
    "dinov3_l_r224": ("sweep", "dinov3_l", "224"),
}
NSTAGE = {"resnet50_r112": 1, "resnet50_r224": 1,
          "dinov2_l_r64": 5, "dinov2_l_r224": 5, "dinov3_l_r224": 4}


def sum_model(t, label):
    kind, model, res = SUM_OF[label]
    if kind == "orin_latency":
        return float(t["orin_latency"][model]["latency"][res])
    return float(t["orin_vitl"]["resolution_sweep_%s" % model]["by_res"][res]["latency_ms"])


def prep_deploy_ms(res, bbox=224):
    """配備経路の CPU 前処理 (1 個体あたり)。optimal_n.py と同じ出所・同じキーで引く"""
    p = os.path.join(R, "results", "orin", "results_prep_trt10", "prep_timing_deploy_maxn.json")
    d = json.load(open(p))["per_bird"]
    return float(d[str(bbox)][str(res)]["total_cv"])


def main():
    t = json.load(open(TABLES))
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".txt"))
    by_label = {}
    for f in files:
        m = re.match(r"(.+)_rep(\d+)\.txt$", f)
        if not m:
            continue
        by_label.setdefault(m.group(1), []).append(os.path.join(SRC, f))

    out = {"note": "1 枚あたり H2D -> 全段 enqueue -> D2H -> 同期 の wall-clock。"
                   "CPU 前処理と CSV 整形は含まない",
           "env": t.get("measurement_env"), "warmup": 200, "by_config": {}}
    print("%-15s %5s | %8s %8s %8s %8s | %8s %8s" %
          ("構成", "段数", "p50", "p95", "mean", "sd", "合算", "差[ms]"))
    for label in sorted(by_label):
        arrs = [np.loadtxt(p) for p in sorted(by_label[label])]
        allv = np.concatenate(arrs)
        s = sum_model(t, label)
        rec = {"n_rep": len(arrs), "n_per_rep": int(len(arrs[0])), "n_total": int(len(allv)),
               "n_stage": NSTAGE[label],
               "p50_ms": float(np.percentile(allv, 50)),
               "p90_ms": float(np.percentile(allv, 90)),
               "p95_ms": float(np.percentile(allv, 95)),
               "p99_ms": float(np.percentile(allv, 99)),
               "mean_ms": float(allv.mean()), "sd_ms": float(allv.std()),
               "min_ms": float(allv.min()), "max_ms": float(allv.max()),
               "rep_p50_ms": [float(np.percentile(a, 50)) for a in arrs],
               "sum_of_parts_ms": s,
               "delta_p50_ms": float(np.percentile(allv, 50)) - s,
               "delta_p95_ms": float(np.percentile(allv, 95)) - s,
               "ratio_p50": float(np.percentile(allv, 50)) / s}
        # ⭐ 配備経路の e2e は CPU 前処理を足したもの。論文の表と同じ足し方にする
        res = int(label.rsplit("_r", 1)[1])
        pre = prep_deploy_ms(res)
        rec["prep_deploy_ms"] = pre
        rec["e2e_deploy_sum_ms"] = pre + s                       # 論文の値 (合算モデル)
        rec["e2e_deploy_wall_p50_ms"] = pre + rec["p50_ms"]      # 実測 wall-clock 版
        rec["e2e_deploy_wall_p95_ms"] = pre + rec["p95_ms"]
        out["by_config"][label] = rec
        print("%-15s %5d | %8.3f %8.3f %8.3f %8.3f | %8.3f %+8.3f" %
              (label, rec["n_stage"], rec["p50_ms"], rec["p95_ms"], rec["mean_ms"],
               rec["sd_ms"], s, rec["delta_p50_ms"]))

    # 段間の上乗せが段数とともに増えるかどうか (段間コストの推定)
    per_stage = [(v["n_stage"], v["delta_p50_ms"]) for v in out["by_config"].values()]
    multi = [d for ns, d in per_stage if ns > 1]
    single = [d for ns, d in per_stage if ns == 1]
    out["overhead"] = {
        "single_stage_delta_p50_ms": single,
        "multi_stage_delta_p50_ms": multi,
        "max_abs_delta_p50_ms": max(abs(v["delta_p50_ms"]) for v in out["by_config"].values()),
        "max_ratio_p95_over_sum": max(v["p95_ms"] / v["sum_of_parts_ms"]
                                      for v in out["by_config"].values()),
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("\n-> %s" % OUT)
    print("   合算との差 (p50): 単段 %s / 多段 %s" %
          (["%.3f" % d for d in single], ["%.3f" % d for d in multi]))


if __name__ == "__main__":
    main()
