#!/usr/bin/env python3
"""中心的な構成の精度に**区間**を付ける (査読指摘 A-9).

論文は 30 シード平均とレイテンシの変動係数しか報告しておらず，test 精度の区間推定が
無かった．ここでは**2 つの変動要因を混ぜずに**別々に出す:

  seed 変動   : 同じデータで学習を引き直したときのばらつき (30 シードの標準偏差)
  画像側変動  : 別の鳥画像を引いたときのばらつき

⚠️⚠️ 画像側は**1 枚単位で再標本化してはいけない**。同一の元写真から切ったクロップや
   同一動画の隣接フレームは独立ではないので，**群 (元画像の機材 ID・観測 ID・動画系列)
   を単位に再標本化する** (cluster bootstrap)。
⚠️ 1,882 枚 x 30 シード を 56,460 の独立標本として数えることはしない。
   画像側は**シード平均した 1 枚あたりの正誤**に対して群を再標本化する。

⭐ 群規則は 2 つある (再レビュー指摘 (1))。--group-rule で選ぶ。

  v1 (既定・従来値の再現用)
      make_group_split.py:group_of と同一。動画フレームは `frame_<n>_bb<n>_...jpg` を
      「種ごとに 1 群」へまとめる規則だが、**`<動画ID8桁>__frame_...jpg` という名前の
      1,000 枚 (bird_frames_pure_oowashi_20260504/) に一致せず 1 枚 1 群**になっていた。
      すなわち同一動画の隣接フレームが独立標本として数えられており、区間が狭く出る。

  v2 (動画 ID を拾う修正規則)
      data/manifest.csv の `group_id_v2` 列 (make_data_manifest.py:group_v2 が生成)を
      rel_path で結合して使う。接頭辞 8 桁の動画 ID ごとに 1 群へまとめるので、
      動画由来の依存が群として正しく効く。

⚠️ v1 の出力 (results/uncertainty.json) は論文が引用済みなので**上書きしない**。
   v2 は --out で別ファイルへ出す。

usage:
  python3 train/uncertainty.py [--boot 2000] [--out results/uncertainty.json]
  python3 train/uncertainty.py --group-rule v2 --out results/uncertainty_groupv2.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "train"))
from make_group_split import group_of  # noqa: E402

PREDS = os.path.join(R, "results", "preds_30seed")
PREDS_VITL = os.path.join(R, "results", "preds_vitl_30seed")
SPLIT_CSV = os.path.join(R, "data", "splits", "test.csv")
MANIFEST = os.path.join(R, "data", "manifest.csv")
RNG_SEED = 20260908

# 本文で中心になる構成 (表 6 の選択例・10 ms 予算・モデル変更の比較)
CONFIGS = [
    ("resnet50", 112), ("resnet50", 224),
    ("vit_small", 224),
    ("dinov2_l", 64), ("dinov2_l", 112), ("dinov2_l", 144),
]
# 本文が述べる差 (a - b)
DIFFS = [
    (("dinov2_l", 112), ("resnet50", 112), "モデル変更 (同一 N=112)"),
    (("resnet50", 224), ("resnet50", 112), "解像度 112->224 (同一モデル)"),
    (("dinov2_l", 64), ("resnet50", 224), "10 ms 予算の 2 構成 (配備 / オフライン)"),
    (("resnet50", 224), ("vit_small", 224), "目標 0.93 付近の当事者"),
]
VITL = ("dinov2_l", "dinov3_l")


def load_manifest_groups(rows):
    """data/manifest.csv の group_id_v2 を rel_path で結合する (群規則 v2).

    ⚠️ 見つからない行があったら黙って落とさず即エラー。1 枚でも欠けると
       群の単位が変わり区間が意味を失う。
    """
    if not os.path.exists(MANIFEST):
        raise SystemExit("[abort] %s が無い。先に train/make_data_manifest.py を実行すること" % MANIFEST)
    man = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            man[r["rel_path"]] = r
    missing = [r["rel_path"] for r in rows if r["rel_path"] not in man]
    if missing:
        raise SystemExit("[abort] test の %d 枚が manifest.csv に無い (例: %s)"
                         % (len(missing), missing[0]))
    # split_image との整合 (test.csv と manifest が同じ分割を指しているか)
    not_test = [r["rel_path"] for r in rows if man[r["rel_path"]]["split_image"] != "test"]
    n_man_test = sum(1 for v in man.values() if v["split_image"] == "test")
    print("manifest.csv %d 行と結合: test %d 枚すべてが一致 (欠損 0)" % (len(man), len(rows)))
    print("  manifest の split_image=='test' は %d 枚 / test.csv は %d 枚 %s"
          % (n_man_test, len(rows), "(一致)" if n_man_test == len(rows) else "⛔ 不一致"))
    if not_test or n_man_test != len(rows):
        raise SystemExit("[abort] manifest の split_image と test.csv が食い違う (%d 枚)" % len(not_test))
    bad_sp = [r["rel_path"] for r in rows if man[r["rel_path"]]["species_key"] != r["species_key"]]
    if bad_sp:
        raise SystemExit("[abort] species_key が manifest と食い違う (%d 枚)" % len(bad_sp))
    return np.array([man[r["rel_path"]]["group_id_v2"] for r in rows])


def load_labels_groups(rule="v1"):
    lab = np.load(os.path.join(PREDS, "labels.npy"))
    rows = list(csv.DictReader(open(SPLIT_CSV)))
    if len(rows) != len(lab):
        raise SystemExit("[abort] test.csv %d 行と labels %d 件が一致しない" % (len(rows), len(lab)))
    csv_lab = np.array([int(r["species_idx"]) for r in rows])
    if not (csv_lab == lab).all():
        raise SystemExit("[abort] test.csv のラベルと labels.npy が一致しない (並び順が違う)")
    if rule == "v1":
        groups = np.array([group_of(r["rel_path"], r["species_key"]) for r in rows])
    elif rule == "v2":
        groups = load_manifest_groups(rows)
    else:
        raise SystemExit("[abort] 未知の群規則: %s" % rule)
    return lab, groups


def seeds_of(model):
    d = PREDS_VITL if model in VITL else PREDS
    return sorted(int(x[1:]) for x in os.listdir(d)
                  if x.startswith("s") and x[1:].isdigit())


def load_correct(model, res, seeds, lab):
    """(n_seed, n_img) の正誤行列。無いシードは飛ばし、実際に読めたシードも返す"""
    d = PREDS_VITL if model in VITL else PREDS
    out, used = [], []
    for s in seeds:
        path = os.path.join(d, "s%d" % s, "%s_r%d.csv" % (model, res))
        if not os.path.exists(path):
            continue
        pred = np.loadtxt(path, delimiter=",", skiprows=1, usecols=-1, dtype=int)
        if len(pred) != len(lab):
            raise SystemExit("[abort] %s の行数が %d (期待 %d)" % (path, len(pred), len(lab)))
        out.append((pred == lab).astype(float))
        used.append(s)
    if not out:
        raise SystemExit("[abort] %s_r%d の CSV が 1 つも無い" % (model, res))
    return np.array(out), used


def cluster_boot(per_img, groups, rng, B):
    """群を単位に再標本化した平均の分布。per_img は 1 枚あたりの値 (シード平均の正誤)"""
    uniq, inv = np.unique(groups, return_inverse=True)
    by_g = [np.where(inv == i)[0] for i in range(len(uniq))]
    sums = np.array([per_img[ix].sum() for ix in by_g])
    cnts = np.array([len(ix) for ix in by_g])
    ng = len(uniq)
    pick = rng.integers(0, ng, size=(B, ng))
    num = sums[pick].sum(axis=1)
    den = cnts[pick].sum(axis=1)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--group-rule", dest="group_rule", default="v1", choices=["v1", "v2"],
                    help="群の定義。v1=make_group_split.py と同一 (従来値の再現)、"
                         "v2=manifest.csv の group_id_v2 (動画 ID を拾う修正規則)")
    ap.add_argument("--out", default=os.path.join(R, "results", "uncertainty.json"))
    args = ap.parse_args()

    print("群規則: %s" % args.group_rule)
    lab, groups = load_labels_groups(args.group_rule)
    rng = np.random.default_rng(RNG_SEED)
    uniq_g, inv_g = np.unique(groups, return_inverse=True)
    n_group = len(uniq_g)
    sizes = np.bincount(inv_g)
    n_video_group = int(sum(1 for g in uniq_g if "/VIDEO" in g))
    n_video_img = int(sum(sizes[i] for i, g in enumerate(uniq_g) if "/VIDEO" in g))
    print("test %d 枚 / 群 %d 個 (最大の群は %d 枚)" % (len(lab), n_group, sizes.max()))
    print("  うち動画由来の群: %d 個 (%d 枚)  ← 群規則 %s"
          % (n_video_group, n_video_img, args.group_rule))
    print("cluster bootstrap %d 回・乱数シード %d\n" % (args.boot, RNG_SEED))

    out = {"note": "seed 変動と画像側 (群単位 cluster bootstrap) を分けて報告する",
           "group_rule": args.group_rule,
           "group_rule_desc": ("make_group_split.py:group_of と同一 (動画 ID 接頭辞つき"
                               "フレームを拾えず 1 枚 1 群になる)" if args.group_rule == "v1"
                               else "data/manifest.csv の group_id_v2 (動画 ID 8 桁で 1 群)"),
           "n_test": int(len(lab)), "n_group": int(n_group),
           "max_group_size": int(sizes.max()),
           "n_video_group": n_video_group, "n_video_image": n_video_img,
           "n_boot": args.boot, "rng_seed": RNG_SEED, "by_config": {}, "diffs": []}

    cache = {}
    print("%-16s %5s %6s %8s %8s | %-22s" %
          ("構成", "seed", "平均", "seed_sd", "seed_se", "群 bootstrap 95%"))
    for m, r in CONFIGS:
        C, used = load_correct(m, r, seeds_of(m), lab)
        cache[(m, r)] = (C, used)
        acc_seed = C.mean(axis=1)
        per_img = C.mean(axis=0)
        b = cluster_boot(per_img, groups, rng, args.boot)
        lo, hi = np.percentile(b, [2.5, 97.5])
        rec = {"n_seed": len(used), "seeds": [used[0], used[-1]],
               "acc_mean": float(acc_seed.mean()), "acc_sd_seed": float(acc_seed.std(ddof=1)),
               "acc_se_seed": float(acc_seed.std(ddof=1) / np.sqrt(len(used))),
               "group_boot_lo": float(lo), "group_boot_hi": float(hi),
               "group_boot_halfwidth_pt": float((hi - lo) / 2 * 100)}
        out["by_config"]["%s_r%d" % (m, r)] = rec
        print("%-16s %5d %6.4f %8.4f %8.4f | [%.4f, %.4f] (±%.2f pt)"
              % ("%s N=%d" % (m, r), len(used), rec["acc_mean"], rec["acc_sd_seed"],
                 rec["acc_se_seed"], lo, hi, rec["group_boot_halfwidth_pt"]))

    print("\n差 [ポイント] (共通シードで対応をとる。群 bootstrap は 1 枚ごとの差に対して行う)")
    print("%-42s %8s %8s | %-24s" % ("比較", "差", "seed_sd", "群 bootstrap 95%"))
    for a, b_, label in DIFFS:
        for k in (a, b_):
            if k not in cache:
                C, used = load_correct(k[0], k[1], seeds_of(k[0]), lab)
                cache[k] = (C, used)
        Ca, sa = cache[a]
        Cb, sb = cache[b_]
        common = sorted(set(sa) & set(sb))          # ViT-L は 29 シードなので共通で揃える
        ia = [sa.index(s) for s in common]
        ib = [sb.index(s) for s in common]
        d_seed = (Ca[ia].mean(axis=1) - Cb[ib].mean(axis=1)) * 100
        per_img = (Ca[ia].mean(axis=0) - Cb[ib].mean(axis=0))
        bs = cluster_boot(per_img, groups, rng, args.boot) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec = {"a": "%s_r%d" % a, "b": "%s_r%d" % b_, "label": label,
               "n_common_seed": len(common),
               "diff_pt": float(d_seed.mean()), "sd_seed_pt": float(d_seed.std(ddof=1)),
               "group_boot_lo_pt": float(lo), "group_boot_hi_pt": float(hi),
               "excludes_zero": bool(lo > 0 or hi < 0)}
        out["diffs"].append(rec)
        print("%-42s %+8.2f %8.2f | [%+.2f, %+.2f]%s"
              % (label + " (%d seed)" % len(common), rec["diff_pt"], rec["sd_seed_pt"],
                 lo, hi, "  0 を含まない" if rec["excludes_zero"] else "  ⚠ 0 を含む"))

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
