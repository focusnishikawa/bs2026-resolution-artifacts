#!/usr/bin/env python3
"""シード数 k と結論の安定性 (査読指摘 B-A2).

「3 シードでは符号が一致して見えるが 30 シードでは割れる」という主張は、
**最初の 3 本 (seed 42-44) だけ**を 30 本と比べたものだった。都合のよい 3 本を
選んだのではないかという疑いが残るので、**30 本から k 本を選ぶ全組合せ**
(k=3 なら 30C3 = 4,060 通り) を評価し直す。

出力する量:

  agree(k)  部分集合の平均の符号が、**30 シード平均の符号**と一致する割合
  unan(k)   部分集合の k 本が**全部同じ符号**になる割合 (「一貫した効果に見える」割合)
  unan_wrong(k)
            全部同じ符号になり、**かつその符号が 30 シード平均と逆**になる割合

⚠️ これは推測統計ではない。**符号の安定性の記述**であって「有意差」ではない。

usage:
  python3 train/seed_stability.py [--out results/seed_stability.json]
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]
LABEL = {"mnv4": "MobileNetV4", "effb0": "EfficientNet-B0", "resnet50": "ResNet50",
         "vit_small": "ViT-S/16", "dinov2_l": "DINOv2-L", "dinov3_l": "DINOv3-L"}
# ⚠️ N=224 は条件 A と条件 B が同一の入力になるので差が定義できない。必ず除く
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208]
KS = [1, 3, 5, 10, 20, 30]
MAX_ENUM = 200000      # これを超える組合せ数は無作為抽出に切り替える
N_SAMPLE = 200000
RNG_SEED = 20260905    # ⚠️ 抽出は固定シード。再実行で値が変わらないようにする
BIG_EFFECT_PT = 1.0    # |30 シード平均差| がこれ以上を「効果が明瞭な組」とする


def subset_index(n, k, rng):
    """(n 本から k 本) の添字行列。全列挙できるならする、無理なら無作為抽出."""
    from math import comb
    total = comb(n, k)
    if total <= MAX_ENUM:
        return np.array(list(combinations(range(n), k)), dtype=np.int64), total, True
    idx = np.empty((N_SAMPLE, k), dtype=np.int64)
    for i in range(N_SAMPLE):
        idx[i] = rng.choice(n, size=k, replace=False)
    return idx, total, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(R, "results", "summary_v2.json"))
    ap.add_argument("--out", default=os.path.join(R, "results", "seed_stability.json"))
    args = ap.parse_args()

    s = json.load(open(args.summary))
    seeds = s["seeds"]
    n = len(seeds)
    rng = np.random.default_rng(RNG_SEED)

    # ---- 各 (モデル, 解像度) の条件間差 d = B - A をシードごとに作る ----
    pairs = []
    for m in MODELS:
        for r in RES:
            rec = s["models"][m][str(r)]
            a = np.asarray(rec["A"]["per_seed"], dtype=float)
            b = np.asarray(rec["B"]["per_seed"], dtype=float)
            if a.size != n or b.size != n:
                raise SystemExit("[abort] %s N=%d のシード数が %d でない" % (m, r, n))
            pairs.append({"model": m, "res": r, "d": (b - a) * 100.0})   # ポイント
    print("対象: %d 組 (%d モデル x %d 水準)・%d シード (%d-%d)"
          % (len(pairs), len(MODELS), len(RES), n, seeds[0], seeds[-1]))
    print("⚠️ N=224 は条件 A と B が同一入力になるため除外している\n")

    # ---- 添字行列は k ごとに 1 回だけ作って全組で使い回す ----
    idx_by_k = {}
    for k in KS:
        idx, total, full = subset_index(n, k, rng)
        idx_by_k[k] = (idx, total, full)
        print("  k=%2d: %s %d 通り" % (k, "全列挙" if full else "無作為抽出 %d / " % len(idx), total))
    print()

    out = {"note": "シード数 k と条件間差の符号の安定性。推測統計ではなく符号の記述である",
           "n_seed": n, "seeds": [seeds[0], seeds[-1]], "resolutions": RES,
           "excluded": "N=224 (条件 A と B が同一入力)",
           "rng_seed": RNG_SEED, "max_enumerate": MAX_ENUM, "n_sample": N_SAMPLE,
           "big_effect_pt": BIG_EFFECT_PT, "by_k": {}, "by_pair": []}

    for p in pairs:
        d = p["d"]
        ref = float(d.mean())
        p["ref_mean"] = ref
        p["ref_sign"] = 1 if ref > 0 else -1
        p["big"] = abs(ref) >= BIG_EFFECT_PT

    per_pair_by_k = {}
    for k in KS:
        idx, total, full = idx_by_k[k]
        rows = []
        per_pair_by_k[k] = rows                   # ペア別 CSV 用に取っておく
        pool_u = 0; pool_uw = 0                   # 全ペアを合算した分子・分母
        for p in pairs:
            sub = p["d"][idx]                     # (n_subset, k)
            mean = sub.mean(axis=1)
            agree = float(np.mean(np.sign(mean) == p["ref_sign"]))
            # ⛔⛔ 査読指摘 A-2 (2026-09-08): **ゼロを負に混ぜない**。
            #   以前は `pos == 0` を「全部負」と数えていたが、78 組 x 30 シードには
            #   **差が厳密にゼロの値が 34 個** (23 組) ある。「負とゼロの混在」は全一致ではない。
            #   ここでは **「全員正」または「全員負」を全一致**とする (ゼロが 1 つでも
            #   入れば全一致にしない)。従来定義の値も unan_old として残し、差を確認できるようにする。
            pos = (sub > 0).sum(axis=1)
            neg = (sub < 0).sum(axis=1)
            unan = (pos == k) | (neg == k)
            sign_unan = np.where(pos == k, 1, -1)
            unan_wrong = unan & (sign_unan != p["ref_sign"])
            unan_old = (pos == k) | (pos == 0)     # 旧定義 (ゼロを負扱い)
            nu = int(unan.sum())
            pool_u += nu; pool_uw += int(unan_wrong.sum())
            rows.append({"model": p["model"], "res": p["res"], "big": p["big"],
                         "agree": agree,
                         "unan": float(np.mean(unan)),
                         "unan_old": float(np.mean(unan_old)),
                         "unan_wrong": float(np.mean(unan_wrong)),
                         # ⭐ 条件付き割合 P(逆符号 | 全一致)。本文が「そのうち」と書くのはこちら
                         "cond_wrong": (float(unan_wrong.sum()) / nu) if nu else None})
        ag = np.array([r["agree"] for r in rows])
        un = np.array([r["unan"] for r in rows])
        uw = np.array([r["unan_wrong"] for r in rows])
        big = np.array([r["big"] for r in rows])
        rec = {"n_subset_total": total, "enumerated": full,
               "agree_median": float(np.median(ag)),
               "agree_q1": float(np.percentile(ag, 25)),
               "agree_q3": float(np.percentile(ag, 75)),
               "agree_min": float(ag.min()),
               "agree_mean_over_pairs": float(ag.mean()),
               "agree_median_big": float(np.median(ag[big])) if big.any() else None,
               "agree_median_small": float(np.median(ag[~big])) if (~big).any() else None,
               "agree_min_big": float(ag[big].min()) if big.any() else None,
               "agree_min_small": float(ag[~big].min()) if (~big).any() else None,
               "agree_n_all30_big": int((ag[big] == 1.0).sum()) if big.any() else None,
               "unanimous_median": float(np.median(un)),
               "unanimous_mean_over_pairs": float(un.mean()),
               "unanimous_wrong_mean_over_pairs": float(uw.mean()),
               "unanimous_wrong_max": float(uw.max()),
               # ⭐ 査読指摘 A-2: 同時事象 P(全一致 かつ 逆) と条件付き P(逆 | 全一致) を分ける
               "cond_wrong_pooled": (pool_uw / pool_u) if pool_u else None,
               "cond_wrong_mean_over_pairs": float(np.nanmean(
                   np.array([r["cond_wrong"] if r["cond_wrong"] is not None else np.nan
                             for r in rows]))),
               "cond_wrong_max": float(np.nanmax(
                   np.array([r["cond_wrong"] if r["cond_wrong"] is not None else np.nan
                             for r in rows]))),
               # 旧定義 (ゼロを負扱い) との差。論文では使わないが、値の変化を追えるように残す
               "unanimous_median_zero_as_neg": float(np.median(
                   np.array([r["unan_old"] for r in rows])))}
        out["by_k"][str(k)] = rec

    # ⭐ 査読指摘 A-2: 厳密ゼロの実数を記録する (「負とゼロの混在」を全一致にしないため)
    D = np.array([p["d"] for p in pairs])
    out["n_zero_diff"] = int((D == 0).sum())
    out["n_pair_with_zero"] = int((D == 0).any(axis=1).sum())
    out["unanimity_rule"] = "全員正 または 全員負 (ゼロが 1 つでも入れば全一致としない)"

    out["by_pair"] = [{"model": p["model"], "res": p["res"],
                       "mean_diff_pt": p["ref_mean"], "big": bool(p["big"]),
                       "sd_pt": float(p["d"].std(ddof=1)),
                       "n_pos": int((p["d"] > 0).sum()), "n_neg": int((p["d"] < 0).sum()),
                       "n_zero": int((p["d"] == 0).sum())} for p in pairs]
    out["n_big"] = int(sum(1 for p in pairs if p["big"]))
    out["n_small"] = int(sum(1 for p in pairs if not p["big"]))

    # ⭐ 査読指摘 B-2「78 ペア別の分子・分母を補足へ出す」: 要約値だけから推測させない
    csv_path = os.path.splitext(args.out)[0] + "_by_pair.csv"
    with open(csv_path, "w") as f:
        f.write("model,res,mean_diff_pt,sd_pt,big,n_pos,n_neg,n_zero,"
                + ",".join("agree_k%d,unan_k%d,unan_wrong_k%d" % (k, k, k) for k in KS) + "\n")
        for j, p in enumerate(pairs):
            cells = []
            for k in KS:
                r = per_pair_by_k[k][j]
                cells += ["%.6f" % r["agree"], "%.6f" % r["unan"], "%.6f" % r["unan_wrong"]]
            f.write("%s,%d,%.6f,%.6f,%d,%d,%d,%d,%s\n"
                    % (p["model"], p["res"], p["ref_mean"], p["d"].std(ddof=1), int(p["big"]),
                       int((p["d"] > 0).sum()), int((p["d"] < 0).sum()), int((p["d"] == 0).sum()),
                       ",".join(cells)))
    print("ペア別の内訳: %s" % csv_path)

    # ---- 表示 ----
    print("=== シード数 k と符号の安定性 (78 組の中央値) ===")
    print("%3s %10s %10s %10s %10s %10s" % ("k", "一致率", "四分位範囲", "全一致率", "全一致かつ逆", "最悪の組"))
    for k in KS:
        r = out["by_k"][str(k)]
        print("%3d %9.1f%% %4.1f-%4.1f%% %9.1f%% %9.2f%% %9.1f%%"
              % (k, 100 * r["agree_median"], 100 * r["agree_q1"], 100 * r["agree_q3"],
                 100 * r["unanimous_median"], 100 * r["unanimous_wrong_mean_over_pairs"],
                 100 * r["agree_min"]))
    print()
    print("効果量で分けた一致率の中央値 (|30 シード平均差| >= %.1f pt を「明瞭」とする)" % BIG_EFFECT_PT)
    print("  明瞭な組 %d 件 / ほぼ 0 の組 %d 件" % (out["n_big"], out["n_small"]))
    print("%3s %12s %12s" % ("k", "明瞭", "ほぼ0"))
    for k in KS:
        r = out["by_k"][str(k)]
        print("%3d %11.1f%% %11.1f%%" % (k, 100 * r["agree_median_big"], 100 * r["agree_median_small"]))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("\n書き出し: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
