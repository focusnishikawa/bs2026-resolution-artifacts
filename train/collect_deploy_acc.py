#!/usr/bin/env python3
"""Orin 実機の**配備精度**を 30 シードで集計する (査読指摘 A1 への対応).

背景:
  論文の表 5 は **サーバ FP32 の 30 シード平均精度**と **Orin FP16 の速度**を
  組み合わせていた。しかし配備するのは Orin の FP16 エンジンであり、seed 42 の実測で
  ViT-S/16 は N>=96 で FP16 化により最大 -10.4 ポイント劣化する (N=112: 0.8932 -> 0.8427)。
  そこで **配備するエンジンそのものの精度**を 30 シードで測り、報告値をこちらへ差し替える。

入力: Orin の全数推論 CSV  <preds>/s<seed>/<model>_r<res>.csv  (idx,l0..l5,argmax)
      正解ラベル          <labels>  (labels.npy。test は 1,882 枚 / val は 1,876 枚)
出力: results/summary_deploy.json      — **results/summary_v2.json と同じ構造**なので
      optimal_n.py / analyze_all.py に --report-acc / --select-acc で差し込める

⚠️ 欠損を黙って飛ばさない。構成ごとに何シード集まったかを必ず出し、
   足りないものは stdout に警告して終了コード 1 を返す。

⚠️ 壊れた FP16 エンジンは入力によらず**定数を返す**。精度としては無効だが、
   「壊れた」という事実は結果なので集計には含め、`n_const` に本数を記録して警告する。

条件について:
  出力の "B" キーは **条件 B (解像度別ネイティブ学習)** を指す。Orin で測っているのは
  条件 B のエンジンだけなので条件 A は空である (summary_v2.json と同じキー構造を保つため
  "B" に入れている)。条件 A は §4 の記述統計にのみ使い構成選択には用いない。

usage:
  # test (報告用)
  python3 train/collect_deploy_acc.py --preds results/preds_30seed \
      --labels results/preds_30seed/labels.npy --out results/summary_deploy.json
  # val (構成選択用)
  python3 train/collect_deploy_acc.py --preds results/preds_30seed_val \
      --labels results/preds_30seed_val/labels.npy --split val \
      --out results/summary_deploy_val.json
  # test + ViT-L (査読指摘 A-4。ViT-L だけ別ディレクトリ・29 シード)
  python3 train/collect_deploy_acc.py --preds results/preds_30seed \
      --labels results/preds_30seed/labels.npy \
      --models mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l \
      --model-preds dinov2_l=results/preds_vitl_30seed \
                    dinov3_l=results/preds_vitl_30seed \
      --out results/summary_deploy.json
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 既定は CNN 3 種 + ViT-S/16。ViT-L (dinov2_l / dinov3_l) は分割チェーンで
# 1 シード 126 パートになるため既定では測っていない。
# ⭐ 査読指摘 A-4 で seed 43-71 の 29 本だけ別チェーン (run_vitl_acc_seed.sh) で測った。
#    含めるときは --models に足し、--model-preds で置き場所を渡す:
#      --models mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l \
#      --model-preds dinov2_l=results/preds_vitl_30seed dinov3_l=results/preds_vitl_30seed
#    ⚠️ ViT-L は 29 シード・CNN は 30 シードで**母数が違う**。被覆の判定はモデルごとに行う。
MODELS = ["mnv4", "effb0", "resnet50", "vit_small"]
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]


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


def read_argmax(path, n_expect):
    """全数推論 CSV の argmax 列を読む. 使えないときは (None, 理由) を返す."""
    if not os.path.exists(path):
        return None, "CSV が無い"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "空の CSV"
    if "argmax" not in rows[0]:
        return None, "argmax 列が無い"
    if len(rows) != n_expect:
        return None, "行数不一致 (CSV %d / ラベル %d)" % (len(rows), n_expect)
    return np.array([int(r["argmax"]) for r in rows]), None


def find_seeds(preds_dir):
    """<preds>/s<NN> から seed 番号を拾う."""
    out = []
    for p in sorted(glob.glob(os.path.join(preds_dir, "s*"))):
        m = re.match(r"^s(\d+)$", os.path.basename(p))
        if m and os.path.isdir(p):
            out.append(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV の親ディレクトリ (s<NN>/ を含む)")
    ap.add_argument("--labels", default=None,
                    help="labels.npy (既定は <preds>/labels.npy)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="対象 seed (既定は <preds>/s* から自動検出)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="seed 番号:CSVディレクトリ の形で追加 (例 42:results/preds_trt10)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="対象モデル (既定は CNN 3 種 + ViT-S/16 の 4 つ)。"
                         "ViT-L を含めるときは dinov2_l dinov3_l を足す")
    ap.add_argument("--model-preds", nargs="*", default=[], dest="model_preds",
                    help="MODEL=DIR の形でモデル別に CSV ディレクトリを差し替える "
                         "(例 dinov2_l=results/preds_vitl_30seed)。"
                         "ViT-L は別チェーンで測るので置き場所が違う")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--n_classes", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--require", type=int, default=None,
                    help="この数のシードがそろわない構成があれば警告して rc=1 (既定は検出数)")
    args = ap.parse_args()

    labels_path = args.labels or os.path.join(args.preds, "labels.npy")
    if not os.path.exists(labels_path):
        sys.exit("[abort] labels が無い: %s\n"
                 "  fgpu0 の inputs%s/labels.npy を回収すること"
                 % (labels_path, "_val" if args.split == "val" else ""))
    labels = np.load(labels_path)
    n = len(labels)

    target_models = args.models if args.models else MODELS

    # ⭐ モデルごとに CSV の置き場所が違いうる。ViT-L は別チェーン (run_vitl_acc_seed.sh) で
    #    測るので preds_vitl_30seed/ にあり、seed も 43-71 の 29 本で CNN の 30 本と違う。
    #    --model-preds を渡さなければ全モデルが --preds を使う = 従来と同一の挙動。
    override = {}
    for spec in args.model_preds:
        if "=" not in spec:
            sys.exit("[abort] --model-preds は MODEL=DIR の形で渡すこと: %s" % spec)
        k, v = spec.split("=", 1)
        override[k] = v

    def dirs_for(m):
        """モデル m の seed -> CSV ディレクトリ."""
        base = override.get(m)
        if base is None:
            base = args.preds
            d = {s: os.path.join(base, "s%d" % s)
                 for s in (args.seeds if args.seeds is not None else find_seeds(base))}
            # --extra は既定の置き場所を使うモデルにだけ効かせる
            for spec in args.extra:
                if ":" not in spec:
                    sys.exit("[abort] --extra は seed:dir の形で渡すこと: %s" % spec)
                k, v = spec.split(":", 1)
                d[int(k)] = v
            return d
        return {s: os.path.join(base, "s%d" % s)
                for s in (args.seeds if args.seeds is not None else find_seeds(base))}

    mdirs = {m: dirs_for(m) for m in target_models}
    seeds = sorted({s for d in mdirs.values() for s in d})
    if not seeds:
        sys.exit("[abort] seed が 1 つも見つからない: %s" % args.preds)

    print("=== Orin 配備精度 (%s, n=%d, %d シード) ===" % (args.split, n, len(seeds)))
    print("    ラベル: %s" % labels_path)
    print("    seed  : %s" % (", ".join(str(x) for x in seeds)))
    for m in target_models:
        if m in override:
            print("    %-10s %s (%d シード)" % (m, override[m], len(mdirs[m])))

    models, short, const_tags = {}, [], []
    for m in target_models:
        models[m] = {}
        dirs = mdirs[m]
        m_seeds = sorted(dirs)
        for r in RES:
            accs, mrs, n_const = [], [], 0
            for s in m_seeds:
                p = os.path.join(dirs[s], "%s_r%d.csv" % (m, r))
                pred, why = read_argmax(p, n)
                if pred is None:
                    continue
                if len(np.unique(pred)) <= 1:
                    # 壊れた FP16 エンジンは定数を返す。事実として数えるが警告する。
                    n_const += 1
                    const_tags.append("s%d/%s_r%d" % (s, m, r))
                accs.append(float((pred == labels).mean()))
                mrs.append(macro_recall(pred, labels, args.n_classes))
            if not accs:
                short.append(("%s_r%d" % (m, r), 0, len(m_seeds)))
                continue
            if len(accs) < len(m_seeds):
                short.append(("%s_r%d" % (m, r), len(accs), len(m_seeds)))
            rec = {"per_seed": accs, "mean": mean(accs), "std": std(accs),
                   "macro_recall_mean": mean(mrs), "n_seed": len(accs)}
            if n_const:
                rec["n_const"] = n_const
            models[m][str(r)] = {"B": rec}

    # ⚠️ ViT-L を含めたときは note を変える。「ViT-L は対象外」と書いたまま
    #    ViT-L が入っていると、読んだ人が中身を誤解する。
    vitl = [m for m in target_models if m in ("dinov2_l", "dinov3_l")]
    tail = ("ViT-L は対象外" if not vitl else
            "ViT-L (%s) は別チェーンで測った %d シード分を含む (査読指摘 A-4)"
            % (", ".join(vitl), max(len(mdirs[m]) for m in vitl)))
    out = {"metric": "acc", "split": args.split, "precision": "orin_fp16_deployed",
           "note": ("Orin Nano の配備エンジン (FP16) を実機で全数推論した精度。"
                    "サーバ PyTorch FP32 の精度ではない (査読指摘 A1)。"
                    "条件 B (解像度別ネイティブ学習) のみ。" + tail),
           "seeds": seeds, "resolutions": RES, "models": models,
           "labels": os.path.abspath(labels_path), "n": int(n)}
    if override:
        out["seeds_by_model"] = {m: sorted(mdirs[m]) for m in target_models}

    print("\n%-12s %s" % ("model", "  ".join("%7d" % r for r in RES)))
    for m in target_models:
        row = []
        for r in RES:
            d = models[m].get(str(r))
            row.append("%7.4f" % d["B"]["mean"] if d else "      -")
        print("%-12s %s" % (m, "  ".join(row)))

    # ⚠️ 期待シード数はモデルごとに違う (ViT-L は 29 本・CNN は 30 本)。
    #    全体の len(seeds) で判定すると ViT-L が永久に「足りない」と出る。
    n_full = sum(1 for m in target_models for r in RES
                 if models[m].get(str(r), {}).get("B", {}).get("n_seed") == len(mdirs[m]))
    if len({len(mdirs[m]) for m in target_models}) == 1:
        print("\n被覆: %d/%d 構成が %d シードそろい"
              % (n_full, len(target_models) * len(RES), len(seeds)))
    else:
        print("\n被覆: %d/%d 構成がモデルごとの期待シード数だけそろい (%s)"
              % (n_full, len(target_models) * len(RES),
                 ", ".join("%s=%d" % (m, len(mdirs[m])) for m in target_models)))
    if short:
        print("⚠️ シードが足りない構成 %d 件:" % len(short))
        for tag, k, need_m in short[:20]:
            print("   %-16s %d / %d シード" % (tag, k, need_m))
        if len(short) > 20:
            print("   ... 他 %d 件" % (len(short) - 20))
    if const_tags:
        print("⛔ 定数出力 (壊れた FP16 エンジン) %d 件: %s"
              % (len(const_tags), ", ".join(const_tags[:10])))
        print("   これらは精度として無効である。集計には含めてあるので必要なら除外すること")
    else:
        print("定数出力: 0 件")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("\n書き出し: %s" % args.out)

    # --require が無ければ「そのモデルで見つかったシード数」を基準にする (従来どおり)
    bad = [t for t, k, need_m in short
           if k < (args.require if args.require is not None else need_m)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
