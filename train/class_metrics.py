#!/usr/bin/env python3
"""選択された構成のクラス別性能 (査読指摘 B-3).

accuracy だけでは少数種の見逃しが見えない。中心となる構成について macro-F1・
balanced accuracy・クラス別再現率・混同の多い種の組を出す。

⚠️ 予測は**配備エンジン**の全数出力 (preds_30seed / preds_vitl_30seed)。
   シードをまたぐときは 1 枚ごとに多数決ではなく、**シードごとに指標を計算して平均**する
   (多数決はアンサンブルになってしまい、配備する 1 本の性能ではなくなる)。

usage: python3 train/class_metrics.py [--out results/class_metrics.json]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "train"))
from uncertainty import CONFIGS, PREDS, PREDS_VITL, VITL, load_correct, seeds_of  # noqa: E402

SPLIT_CSV = os.path.join(R, "data", "splits", "test.csv")


def load_pred(model, res, seeds, n):
    d = PREDS_VITL if model in VITL else PREDS
    out, used = [], []
    for s in seeds:
        path = os.path.join(d, "s%d" % s, "%s_r%d.csv" % (model, res))
        if not os.path.exists(path):
            continue
        p = np.loadtxt(path, delimiter=",", skiprows=1, usecols=-1, dtype=int)
        if len(p) != n:
            raise SystemExit("[abort] %s の行数が %d" % (path, len(p)))
        out.append(p); used.append(s)
    return np.array(out), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "results", "class_metrics.json"))
    args = ap.parse_args()

    lab = np.load(os.path.join(PREDS, "labels.npy"))
    rows = list(csv.DictReader(open(SPLIT_CSV)))
    keys = {}
    for r in rows:
        keys[int(r["species_idx"])] = r["species_key"]
    names = [keys[i] for i in range(len(keys))]
    n_cls = len(names)
    cnt = np.bincount(lab, minlength=n_cls)
    print("test %d 枚 / %d クラス" % (len(lab), n_cls))
    print("クラス別枚数: %s" % ", ".join("%s %d" % (names[i], cnt[i]) for i in range(n_cls)))
    print("最大/最小の比 = %.2f\n" % (cnt.max() / cnt.min()))

    out = {"note": "配備エンジンの全数予測から、シードごとに指標を計算して平均した",
           "n_test": int(len(lab)), "classes": names,
           "support": [int(x) for x in cnt], "by_config": {}}
    print("%-16s %5s %8s %8s %8s | %s" %
          ("構成", "seed", "acc", "macroF1", "balAcc", "最も低いクラス再現率"))
    for m, r in CONFIGS:
        P, used = load_pred(m, r, seeds_of(m), len(lab))
        accs, f1s, bals, recs = [], [], [], []
        conf = np.zeros((n_cls, n_cls))
        for p in P:
            accs.append((p == lab).mean())
            rec = np.array([(p[lab == c] == c).mean() for c in range(n_cls)])
            pre = np.array([(lab[p == c] == c).mean() if (p == c).any() else 0.0
                            for c in range(n_cls)])
            f1 = np.where(rec + pre > 0, 2 * rec * pre / np.where(rec + pre > 0, rec + pre, 1), 0.0)
            f1s.append(f1.mean()); bals.append(rec.mean()); recs.append(rec)
            for a, b in zip(lab, p):
                conf[a, b] += 1
        recs = np.array(recs); conf /= len(P)
        rec_mean = recs.mean(axis=0)
        worst = int(np.argmin(rec_mean))
        off = conf.copy(); np.fill_diagonal(off, 0)
        i, j = np.unravel_index(np.argmax(off), off.shape)
        rec = {"n_seed": len(used), "acc": float(np.mean(accs)),
               "macro_f1": float(np.mean(f1s)), "balanced_acc": float(np.mean(bals)),
               "recall_by_class": {names[c]: float(rec_mean[c]) for c in range(n_cls)},
               "worst_class": names[worst], "worst_recall": float(rec_mean[worst]),
               "acc_minus_macro_f1_pt": float((np.mean(accs) - np.mean(f1s)) * 100),
               "top_confusion": {"true": names[i], "pred": names[j],
                                 "count_mean": float(off[i, j])}}
        out["by_config"]["%s_r%d" % (m, r)] = rec
        print("%-16s %5d %8.4f %8.4f %8.4f | %s %.4f (最多の混同 %s->%s %.1f 枚)"
              % ("%s N=%d" % (m, r), len(used), rec["acc"], rec["macro_f1"],
                 rec["balanced_acc"], rec["worst_class"], rec["worst_recall"],
                 names[i], names[j], off[i, j]))

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
