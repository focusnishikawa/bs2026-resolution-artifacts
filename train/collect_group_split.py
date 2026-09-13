#!/usr/bin/env python3
"""査読指摘 A-3: group split (元画像・動画系列をまたがない分割) の評価を集計する.

背景:
  本編の分割は画像単位なので、同じ元写真から切り出したクロップや同じ動画の隣接フレームが
  train と test に分かれて入りうる。実測では **test 61 枚 (3.24%) / val 58 枚 (3.09%)** が
  漏洩しており、内訳は動画の隣接フレームが 51 枚と支配的である
  (再現: train/make_group_split.py --report)。
  そこで群 (元写真の機材 ID・観測 ID・動画系列) をまたがない分割で境界 16 構成を学習し直し、
  **選択が入れ替わるか**を確認する。レビューの言う「探索的結果 / 確認的結果」の分離である。

入力: results/G_cond{A,B}_s<seed>/<model>_r<res>.json   (eval_sweep.py の出力)
出力: results/summary_group.json                        (results/summary_v2.json と同構造)

⭐ 群規則 v2 (動画 ID を含む群定義) で作り直した系列は入力・出力とも別物である。
   --cond-prefix で入力ディレクトリの接頭辞を差し替える (既定 "G_cond" = 従来どおり)。
     python3 train/collect_group_split.py --cond-prefix G2_cond \
         --out /work/gfsi/ufsi0002/bs2026-resolution/results/summary_group_v2.json

⚠️ 生データは HPC にしか無いので **HPC 上で実行する** (--root を指定すれば Mac でも動く)。
⚠️ 1 件も読めなければ書き出さない。nan で埋まった JSON を作ると後段が全部壊れる
   (collect_v2.py が同じ事故を起こした経緯がある)。
⚠️ 条件 A の DINOv2-L は dinov2_l_r224 が要る。追加学習が済むまでは欠測になるので、
   構成ごとの n_seed を必ず出す。

usage:
  python3 train/collect_group_split.py                      # HPC 上
  python3 train/collect_group_split.py --root . --no-write  # Mac で確認だけ
"""
import argparse
import glob
import json
import os
import re
import sys

# 境界 16 構成 (train/run_group_split_boundary.sh と同じ)
BOUNDARY = {
    "resnet50": [32, 48, 64, 80, 96, 112, 128, 160, 176, 208, 224],
    "dinov2_l": [64, 144, 160],
    "vit_small": [112, 224],
}
MODELS = list(BOUNDARY)
N_BOUNDARY = sum(len(v) for v in BOUNDARY.values())
PAT = re.compile(r"^(resnet50|dinov2_l|vit_small)_r(\d+)\.json$")
COND_PREFIX = "G_cond"      # v1 (A-3 本体)。v2 は G2_cond を --cond-prefix で渡す


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load(mode, seed, root, prefix=COND_PREFIX):
    out = {}
    d = os.path.join(root, "results", "%s%s_s%d" % (prefix, mode, seed))
    for f in glob.glob(os.path.join(d, "*.json")):
        if not PAT.match(os.path.basename(f)):
            continue
        try:
            j = json.load(open(f))
        except (ValueError, OSError) as e:      # 書きかけの JSON を握り潰さない
            print("  ⚠️ 読めない: %s (%s)" % (f, e))
            continue
        out[(j["model"], j["res"])] = j
    return out


def find_seeds(root, prefix=COND_PREFIX):
    seeds = []
    for d in glob.glob(os.path.join(root, "results", "%sB_s*" % prefix)):
        m = re.search(r"_s(\d+)$", d)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/work/gfsi/ufsi0002/bs2026-resolution")
    p.add_argument("--metric", default="acc", choices=["acc", "macro_recall"])
    p.add_argument("--out", default=None, help="既定は <root>/results/summary_group.json")
    p.add_argument("--no-write", action="store_true", dest="no_write")
    p.add_argument("--compare", default=None,
                   help="通常分割の集計 (既定 <root>/results/summary_v2.json)")
    p.add_argument("--cond-prefix", dest="cond_prefix", default=COND_PREFIX,
                   help="入力ディレクトリの接頭辞。既定 %s (v1)。"
                        "群規則 v2 の系列は G2_cond" % COND_PREFIX)
    args = p.parse_args()

    seeds_all = find_seeds(args.root, args.cond_prefix)
    if not seeds_all:
        sys.exit("[中断] results/%sB_s* が 1 つも無い: %s\n"
                 "        本スクリプトは生データのある HPC 上で実行すること。"
                 % (args.cond_prefix, args.root))

    # ⚠️ 条件 B が 16 構成そろった seed だけを平均に使う。中途半端な seed を混ぜると
    #    水準ごとに母数が変わり、比較そのものが壊れる (collect_v2.py と同じ方針)。
    data = {"A": {}, "B": {}}
    seeds, partial = [], []
    for s in seeds_all:
        b = load("B", s, args.root, args.cond_prefix)
        a = load("A", s, args.root, args.cond_prefix)
        data["B"][s], data["A"][s] = b, a
        if len(b) >= N_BOUNDARY:
            seeds.append(s)
        else:
            partial.append((s, len(b), len(a)))

    print("=== group split の集計 (%s) ===" % args.metric)
    print("  root : %s" % args.root)
    print("  seed : 条件 B が %d 構成そろったもの %d 本 %s"
          % (N_BOUNDARY, len(seeds), seeds if len(seeds) <= 32 else "..."))
    if partial:
        print("  ⚠️ そろっていない seed %d 本 (平均から除外):" % len(partial))
        for s, nb, na in partial[:10]:
            print("     s%-3d 条件B %2d/%d ・条件A %2d/%d" % (s, nb, N_BOUNDARY, na, N_BOUNDARY))
        if len(partial) > 10:
            print("     ... 他 %d 本" % (len(partial) - 10))
    if not seeds:
        sys.exit("[中断] 条件 B が %d 構成そろった seed が 1 つも無い。書き出さない。" % N_BOUNDARY)

    summary = {"metric": args.metric, "split": "test", "seeds": seeds,
               "note": ("group split (元写真の機材 ID・観測 ID・動画系列をまたがない分割) で"
                        "境界 16 構成を学習し直した結果 (査読指摘 A-3)。"
                        "サーバ PyTorch FP32・条件 A/B。本編の画像単位分割とは別物である。"),
               "boundary": {m: BOUNDARY[m] for m in MODELS},
               "resolutions": sorted({r for v in BOUNDARY.values() for r in v}),
               "models": {}}

    for label, mode in [("条件A (劣化耐性)", "A"), ("条件B (解像度別ネイティブ学習)", "B")]:
        print("\n--- %s: %s 平均 (%d seed) ---" % (label, args.metric, len(seeds)))
        for m in MODELS:
            row = ""
            for r in BOUNDARY[m]:
                xs = [data[mode][s][(m, r)][args.metric] for s in seeds if (m, r) in data[mode][s]]
                row += ("%8.4f" % mean(xs)) if xs else "       -"
            print("%-10s %s" % (m, row))

    n_short = 0
    for m in MODELS:
        summary["models"][m] = {}
        for r in BOUNDARY[m]:
            e = {}
            for mode in ["A", "B"]:
                xs = [data[mode][s][(m, r)][args.metric] for s in seeds if (m, r) in data[mode][s]]
                mrs = [data[mode][s][(m, r)]["macro_recall"] for s in seeds if (m, r) in data[mode][s]]
                e[mode] = {"per_seed": xs, "mean": mean(xs), "std": std(xs),
                           "macro_recall_mean": mean(mrs), "n_seed": len(xs)}
                if len(xs) < len(seeds):
                    n_short += 1
            summary["models"][m][str(r)] = e

    if n_short:
        print("\n⚠️ シードが欠けている (構成, 条件) の組が %d 件ある。"
              "条件 A の DINOv2-L は dinov2_l_r224 の追加学習待ちかもしれない" % n_short)
        for m in MODELS:
            for r in BOUNDARY[m]:
                for mode in ["A", "B"]:
                    k = summary["models"][m][str(r)][mode]["n_seed"]
                    if k < len(seeds):
                        print("   cond%s %-10s r%-3d  %d/%d シード" % (mode, m, r, k, len(seeds)))

    # ---- 通常分割との比較 ----
    cmp_path = args.compare or os.path.join(args.root, "results", "summary_v2.json")
    if os.path.exists(cmp_path):
        base = json.load(open(cmp_path))
        common = [s for s in seeds if s in base.get("seeds", [])]
        print("\n=== 通常分割との差 [pt] (group split - 画像単位分割・共通 %d seed) ==="
              % len(common))
        if not common:
            print("  ⚠️ 共通 seed が無いので比較できない")
        else:
            bidx = [base["seeds"].index(s) for s in common]
            for mode in ["A", "B"]:
                print("--- 条件 %s ---" % mode)
                for m in MODELS:
                    row = ""
                    for r in BOUNDARY[m]:
                        g = [data[mode][s][(m, r)][args.metric]
                             for s in common if (m, r) in data[mode][s]]
                        bp = base["models"].get(m, {}).get(str(r), {}).get(mode, {}).get("per_seed")
                        if not g or not bp or len(bp) <= max(bidx):
                            row += "       -"
                            continue
                        b = [bp[i] for i in bidx]
                        row += "%+8.2f" % ((mean(g) - mean(b)) * 100)
                    print("%-10s %s" % (m, row))
            summary["compare"] = {"source": os.path.basename(cmp_path), "common_seeds": common}
    else:
        print("\n(比較対象 %s が無いので差は出さない)" % cmp_path)

    out = args.out or os.path.join(args.root, "results", "summary_group.json")
    if args.no_write:
        print("\n[--no-write] 書き出さない (%s)" % out)
        return 0
    with open(out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("\n[saved] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
