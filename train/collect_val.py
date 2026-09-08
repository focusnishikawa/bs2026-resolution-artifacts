#!/usr/bin/env python3
"""validation の予測から精度を集計する (査読指摘 A5 への対応).

背景:
  84 構成のパレートフロント・精度目標ごとの最短構成・N の選択を **test 集合**で行い、
  同じ集合から最終性能を報告すると選択バイアスが入る。そこで
  **選択は validation、報告は test** に分ける。ここでは選択側の材料を作る。

入力: results/T1_condB_s<seed>_val/preds/<model>_r<res>.npz  (probs, labels, rel_paths)
出力: results/summary_val.json  — results/summary_v2.json と同じ構造なので、
      optimal_n.py / analyze_all.py に --acc_json で差し替えられる

⚠️ 欠損は黙って飛ばさない。構成ごとに何シード集まったかを必ず出し、
   30 に満たないものは stdout に警告する。

usage:
  python3 train/collect_val.py --root /work/gfsi/ufsi0002/bs2026-resolution \
      --out results/summary_val.json
"""
import argparse
import json
import os
import sys

import numpy as np

RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)


def macro_recall(pred, labels, n_cls):
    rs = []
    for c in range(n_cls):
        sel = labels == c
        if sel.sum():
            rs.append(float((pred[sel] == c).mean()))
    return mean(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(42, 72)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_classes", type=int, default=6)
    args = ap.parse_args()

    models = {}
    short = []
    for m in MODELS:
        models[m] = {}
        for r in RES:
            accs, mrs = [], []
            for s in args.seeds:
                p = os.path.join(args.root, "results", "T1_condB_s%d_val" % s,
                                 "preds", "%s_r%d.npz" % (m, r))
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                pred = z["probs"].argmax(1)
                lab = z["labels"]
                accs.append(float((pred == lab).mean()))
                mrs.append(macro_recall(pred, lab, args.n_classes))
            if not accs:
                short.append(("%s_r%d" % (m, r), 0))
                continue
            if len(accs) < len(args.seeds):
                short.append(("%s_r%d" % (m, r), len(accs)))
            models[m][str(r)] = {"B": {"mean": mean(accs), "std": std(accs),
                                       "macro_recall_mean": mean(mrs),
                                       "n_seed": len(accs)}}

    out = {"metric": "acc", "split": "val", "seeds": args.seeds,
           "resolutions": RES, "models": models}

    print("=== validation 精度 (条件 B, 30 シード平均) ===")
    print("%-12s %s" % ("model", "  ".join("%7d" % r for r in RES)))
    for m in MODELS:
        row = []
        for r in RES:
            d = models[m].get(str(r))
            row.append("%7.4f" % d["B"]["mean"] if d else "      -")
        print("%-12s %s" % (m, "  ".join(row)))

    n_full = sum(1 for m in MODELS for r in RES
                 if models[m].get(str(r), {}).get("B", {}).get("n_seed") == len(args.seeds))
    print("\n被覆: %d/%d 構成が %d シードそろい" % (n_full, len(MODELS) * len(RES), len(args.seeds)))
    if short:
        print("⚠️ シードが足りない構成 %d 件:" % len(short))
        for tag, n in short[:20]:
            print("   %-16s %d シード" % (tag, n))
        if len(short) > 20:
            print("   ... 他 %d 件" % (len(short) - 20))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("\n書き出し: %s" % args.out)
    return 0 if not short else 1


if __name__ == "__main__":
    sys.exit(main())
