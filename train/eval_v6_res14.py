#!/usr/bin/env python3
"""Phase 4 の集計: v6 カスケードの解像度スイープを 3 系統まとめて評価する.

対象:
  条件 A 分離版  preds/v6_40k_{flat,hier}_gt_{bird,nonbird}_ds<N>.csv     (既存モデルに縮小入力)
  条件 A 統合版  preds/v6_s34r500_gt_{bird,nonbird}_{ds<N>|native}.csv    (74 クラス 500 木)
  条件 B 統合版  preds/v6_condB_s34r500_gt_{bird,nonbird}_cb<N>.csv       (解像度別に再学習)

段別に別の分母で採点する v6 の流儀を踏襲する:
  Stage1 鳥/非鳥      : 鳥 840 + 非鳥 840 = 1,680 が分母
  Stage2 猛禽/非猛禽  : 鳥 840 が分母
  Stage4 種           : タカ目 GT のみが分母
  Stage5 IUCN         : 1,680 が分母。自明ベースライン (常に OTHERS) も併記する
  保護対象種アラート  : precision / recall を対で読む (片方だけでは意味を持たない)

出力: results/v6_res14_summary.json + 標準出力の表
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

PROTECTED = ["Inuwashi_Golden_Eagle", "Ojirowashi_White_Tailed_Eagle",
             "Oowashi_Stellers_Sea_Eagle", "Kumataka_Mountain_Hawk_Eagle"]
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]


def read_preds(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def load_gt(csv_path):
    """GT の正解ラベルを CSV から読む。

    ⚠️ 画像パスのディレクトリ名から種名を推定してはいけない。GT 840 枚は
    「猛禽 370 + 非猛禽 450 + 空 20」であり、全部を猛禽扱いすると Stage2 が
    0.4452 (=370/840) になって論文値 0.9333 と食い違う。分母は段ごとに違う。
    """
    gt = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            gt[r["image_path"]] = r
    return gt


def evaluate(bird_rows, nonbird_rows, gt_bird, c2t):
    """1 構成ぶんの段別指標。v6 の流儀どおり段ごとに別の分母で採点する。"""
    out = {}
    nb, nn = len(bird_rows or []), len(nonbird_rows or [])
    if not nb or not nn:
        return None

    # Stage1: 鳥/非鳥 (分母 = 鳥 840 + 非鳥 840)
    s1 = sum(1 for r in bird_rows if r["stage1"] == "bird")
    s1 += sum(1 for r in nonbird_rows if r["stage1"] == "nonbird")
    out["stage1_acc"] = s1 / (nb + nn)
    out["nonbird_reject"] = sum(1 for r in nonbird_rows if r["stage1"] == "nonbird") / nn

    # Stage2: 猛禽/非猛禽の 2 値正解率 (分母 = 鳥 840。GT の raptor は 370 のみ)
    ok = tot = 0
    for r in bird_rows:
        g = gt_bird.get(r["image_path"])
        if not g or not g["stage2"]:
            continue
        tot += 1
        ok += int(r.get("stage2") == g["stage2"])
    out["stage2_acc"] = ok / tot if tot else None
    out["stage2_n"] = tot

    # Stage4: 種 (分母 = GT が種名を持つもの = タカ目のみ)
    hit = top5 = tot4 = 0
    for r in bird_rows:
        g = gt_bird.get(r["image_path"])
        if not g or not g["stage4"]:
            continue
        tot4 += 1
        hit += int(r.get("stage4") == g["stage4"])
        top5 += int(g["stage4"] in (r.get("top5_s4") or "").split("|"))
    out["stage4_top1"] = hit / tot4 if tot4 else None
    out["stage4_top5"] = top5 / tot4 if tot4 else None
    out["stage4_n"] = tot4

    # Stage5: IUCN (分母 = 1,680)。非鳥の正解は OTHERS
    ok5 = 0
    for r in bird_rows:
        g = gt_bird.get(r["image_path"])
        ok5 += int(r.get("stage5_iucn") == (g["iucn_gt"] if g else "OTHERS"))
    ok5 += sum(1 for r in nonbird_rows if r.get("stage5_iucn") == "OTHERS")
    out["iucn_acc"] = ok5 / (nb + nn)
    triv = sum(1 for r in bird_rows
               if (gt_bird.get(r["image_path"]) or {}).get("iucn_gt", "OTHERS") == "OTHERS") + nn
    out["iucn_trivial"] = triv / (nb + nn)

    # 保護対象種アラート (precision と recall は必ず対で読む)
    tp = fp = fn = 0
    for r in bird_rows:
        g = gt_bird.get(r["image_path"]) or {}
        pred, truth = r.get("stage4") in PROTECTED, g.get("stage4") in PROTECTED
        tp += int(pred and truth); fp += int(pred and not truth); fn += int(truth and not pred)
    for r in nonbird_rows:
        fp += int(r.get("stage4") in PROTECTED)
    out["alert_precision"] = tp / (tp + fp) if (tp + fp) else None
    out["alert_recall"] = tp / (tp + fn) if (tp + fn) else None
    out["alert_tp"], out["alert_fp"], out["alert_fn"] = tp, fp, fn
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home1/gfsi/ufsi0002/bs2026_v5_rebuild")
    p.add_argument("--out", default="results/v6_res14_summary.json")
    args = p.parse_args()
    R = args.root
    c2t = json.load(open(os.path.join(R, "configs/class_to_taxon_v5.json")))
    gt_bird = load_gt(os.path.join(R, "data/hierarchy_labels_gt_84_y.csv"))
    print("[gt] %d 枚 (raptor %d / non_raptor %d / 種名あり %d)"
          % (len(gt_bird), sum(1 for g in gt_bird.values() if g["stage2"] == "raptor"),
             sum(1 for g in gt_bird.values() if g["stage2"] == "non_raptor"),
             sum(1 for g in gt_bird.values() if g["stage4"])))

    systems = {}
    for n in RES:
        ds = "native" if n == 224 else "ds%d" % n
        # 条件 A 統合版
        systems.setdefault("condA_s34", {})[n] = (
            os.path.join(R, "preds/v6_s34r500_gt_bird_%s.csv" % ds),
            os.path.join(R, "preds/v6_s34r500_gt_nonbird_%s.csv" % ds))
        # 条件 B 統合版 (前段が 224 のままだった旧版。参考として残す)
        systems.setdefault("condB_old", {})[n] = (
            os.path.join(R, "preds/v6_condB_s34r500_gt_bird_cb%d.csv" % n),
            os.path.join(R, "preds/v6_condB_s34r500_gt_nonbird_cb%d.csv" % n))
        # 条件 B 統合版 (Stage1/2 も条件 B で学習し直した修正版)
        systems.setdefault("condB_fix", {})[n] = (
            os.path.join(R, "preds/v6_condBfix_s34r500_gt_bird_cb%d.csv" % n),
            os.path.join(R, "preds/v6_condBfix_s34r500_gt_nonbird_cb%d.csv" % n))
        # 条件 B・ViT-L (代表 4 水準のみ)
        for vm in ("dinov2_l", "dinov3_l"):
            b = os.path.join(R, "preds/v6_vitl_s34r500_gt_bird_%s_r%d.csv" % (vm, n))
            nb2 = os.path.join(R, "preds/v6_vitl_s34r500_gt_nonbird_%s_r%d.csv" % (vm, n))
            if os.path.exists(b):
                systems.setdefault("condB_" + vm, {})[n] = (b, nb2)
        # 条件 A 分離版 (flat)
        suf = "" if n == 224 else "_ds%d" % n
        for cand in ("v6_40k_flat_gt_%s%s.csv", "v6_flat_gt_%s%s.csv"):
            b = os.path.join(R, "preds", cand % ("bird", suf))
            nb_ = os.path.join(R, "preds", cand % ("nonbird", suf))
            if os.path.exists(b):
                systems.setdefault("condA_flat", {})[n] = (b, nb_)
                break

    result = {}
    for sysname, per_res in systems.items():
        result[sysname] = {}
        for n, (bp, np_) in sorted(per_res.items()):
            m = evaluate(read_preds(bp), read_preds(np_), gt_bird, c2t)
            if m:
                result[sysname][str(n)] = m

    os.makedirs(os.path.join(R, "results"), exist_ok=True)
    json.dump(result, open(os.path.join(R, args.out), "w"), indent=2, ensure_ascii=False)

    for metric, label in [("stage1_acc", "Stage1 鳥/非鳥"), ("stage2_acc", "Stage2 猛禽"),
                          ("stage4_top1", "Stage4 種 top-1"), ("stage4_top5", "Stage4 種 top-5"),
                          ("iucn_acc", "Stage5 IUCN"), ("alert_precision", "アラート precision"),
                          ("alert_recall", "アラート recall")]:
        print("\n=== %s ===" % label)
        print("%-12s" % "system" + "".join("%8d" % r for r in RES))
        for sysname in ("condA_flat", "condA_s34", "condB_fix", "condB_dinov2_l", "condB_dinov3_l"):
            if sysname not in result:
                continue
            row = ""
            for r in RES:
                v = result[sysname].get(str(r), {}).get(metric)
                row += ("%8.4f" % v) if v is not None else "       -"
            print("%-12s" % sysname + row)
    print("\n[saved] %s" % os.path.join(R, args.out))


if __name__ == "__main__":
    main()
