#!/usr/bin/env python3
"""サーバ FP32 の validation 集計を **配備側と同じシード集合へ揃える** (再レビュー指摘).

背景 (交絡の指摘):
  論文はサーバ FP32 (`results/summary_val.json`) と配備エンジン
  (`results/summary_deploy_val.json`) の validation を突き合わせているが、

    サーバ側          : 全モデル seed 42-71 の **30 本**平均
    配備側 (ViT-L のみ): seed 43-71 の **29 本**平均
                        (dinov2_l / dinov3_l は 1 シード 126 パートの分割チェーンで
                         測るため seed 42 が無い)

  となっており、ViT-L の「サーバ − 配備」差には **演算精度 (FP32 vs FP16/混合) の影響**と
  **seed 42 を含むか否かの影響**が同時に入っている。この 2 つは分離できない (交絡)。

  そこで**サーバ側の ViT-L だけを seeds 43-71 の 29 本へ落として再集計**し、
  精度差と構成選択 (target sweep) を「シード集合が同一な状態」で出し直す。

  CNN 3 種 (mnv4 / effb0 / resnet50) と vit_small は**両側とも seed 42-71 の 30 本**で
  既に整合しているので、`summary_val.json` の値をそのまま複製する (再計算もしない)。

入力:
  results/summary_val.json                                     現行のサーバ val 集計 (30 シード)
  results/T1_val_perseed/T1_condB_s<seed>_val/preds/<m>_r<N>.npz
      サーバ FP32 の validation 予測 (probs (1876,6) / labels / rel_paths)。
      probs.argmax(1) == labels の平均が per-seed 精度。

出力:
  results/summary_val_aligned.json    summary_val.json と同一構造 (+ per_seed / seeds / meta)

⭐ 回帰検査 (必須・これに通らなければ 29 本集計へ進まない):
  npz から seeds 42-71 の 30 本で集計した値が summary_val.json の mean と 1e-12 以内で
  一致することを ViT-L 全 28 構成 (2 モデル x 14 解像度) で確認する。
  ここが合わないなら npz の読み方か対応付けが違うので、29 本集計をしても意味がない。

usage:
  python3 train/align_server_val.py [--out results/summary_val_aligned.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results")

RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
COPY_MODELS = ["mnv4", "effb0", "resnet50", "vit_small"]   # 両側とも 30 シードで整合済み
ALIGN_MODELS = ["dinov2_l", "dinov3_l"]                    # 配備側に seed 42 が無い
ALL_SEEDS = list(range(42, 72))                            # 42-71 (30 本)
DROP_SEED = 42                                             # 配備側 ViT-L に無いシード
N_CLASSES = 6
PERSEED_DIR = os.path.join(R, "T1_val_perseed")
TOL = 1e-12


# ---- collect_val.py と**同一実装** (足し込みの順序まで揃えてビット一致を狙う) ----
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


def npz_path(seed, model, res):
    return os.path.join(PERSEED_DIR, "T1_condB_s%d_val" % seed, "preds",
                        "%s_r%d.npz" % (model, res))


def per_seed_stats(model, res, seeds):
    """(seed -> acc, seed -> macro_recall). 欠損は黙って飛ばさず中止する"""
    accs, mrs = {}, {}
    for s in seeds:
        p = npz_path(s, model, res)
        if not os.path.exists(p):
            sys.exit("[abort] %s が無い。欠損を黙って飛ばさない" % p)
        z = np.load(p)
        pred = z["probs"].argmax(1)
        lab = z["labels"]
        accs[s] = float((pred == lab).mean())
        mrs[s] = macro_recall(pred, lab, N_CLASSES)
    return accs, mrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(R, "summary_val.json"))
    ap.add_argument("--out", default=os.path.join(R, "summary_val_aligned.json"))
    args = ap.parse_args()

    sv = json.load(open(args.src))
    keep_seeds = [s for s in ALL_SEEDS if s != DROP_SEED]
    assert len(keep_seeds) == 29 and keep_seeds[0] == 43 and keep_seeds[-1] == 71

    print("=== サーバ FP32 val を配備側のシード集合へ揃える ===")
    print("複製 (両側 30 シードで整合済み): %s" % ", ".join(COPY_MODELS))
    print("再集計 (30 -> 29 シード, seed %d を除外): %s" % (DROP_SEED, ", ".join(ALIGN_MODELS)))

    # ------------------------------------------------------------------ 再集計
    raw = {}          # (model, res) -> (accs, mrs)
    for m in ALIGN_MODELS:
        for r in RES:
            raw[(m, r)] = per_seed_stats(m, r, ALL_SEEDS)
    print("npz を読み込み: %d 構成 x %d シード = %d ファイル"
          % (len(raw), len(ALL_SEEDS), len(raw) * len(ALL_SEEDS)))

    # ---- 回帰検査: 30 本集計が summary_val.json と一致するか -------------------
    print("\n--- 回帰検査: npz からの 30 シード集計 vs summary_val.json ---")
    worst = ("", 0.0)
    worst_mr = ("", 0.0)
    worst_std = ("", 0.0)
    for (m, r), (accs, mrs) in sorted(raw.items()):
        ref = sv["models"][m][str(r)]["B"]
        got_mean = mean([accs[s] for s in ALL_SEEDS])
        got_std = std([accs[s] for s in ALL_SEEDS])
        got_mr = mean([mrs[s] for s in ALL_SEEDS])
        d = abs(got_mean - ref["mean"])
        dmr = abs(got_mr - ref["macro_recall_mean"])
        dsd = abs(got_std - ref["std"])
        if d > worst[1]:
            worst = ("%s_r%d" % (m, r), d)
        if dmr > worst_mr[1]:
            worst_mr = ("%s_r%d" % (m, r), dmr)
        if dsd > worst_std[1]:
            worst_std = ("%s_r%d" % (m, r), dsd)
        if ref["n_seed"] != len(ALL_SEEDS):
            sys.exit("[abort] %s_r%d の summary_val.json は n_seed=%d (30 のはず)"
                     % (m, r, ref["n_seed"]))
        assert d <= TOL, ("30 シード平均が summary_val.json と一致しない: %s_r%d "
                          "npz=%.17g json=%.17g diff=%.3g" % (m, r, got_mean, ref["mean"], d))
    print("  acc mean          最大差 %.3g (%s)  <= %g  OK" % (worst[1], worst[0], TOL))
    print("  macro_recall mean 最大差 %.3g (%s)" % (worst_mr[1], worst_mr[0]))
    print("  acc std           最大差 %.3g (%s)" % (worst_std[1], worst_std[0]))
    print("  -> 全 %d 構成でデータ経路が一致。29 シード集計へ進む" % len(raw))

    # ------------------------------------------------------------------ 出力の組み立て
    models = {}
    for m in COPY_MODELS:
        models[m] = json.loads(json.dumps(sv["models"][m]))       # そのまま複製
        for r in RES:
            models[m][str(r)]["B"]["aligned"] = False
            models[m][str(r)]["B"]["seeds"] = list(ALL_SEEDS)
    for m in ALIGN_MODELS:
        models[m] = {}
        accs, mrs = None, None
        for r in RES:
            accs, mrs = raw[(m, r)]
            a29 = [accs[s] for s in keep_seeds]
            m29 = [mrs[s] for s in keep_seeds]
            models[m][str(r)] = {"B": {
                "mean": mean(a29),
                "std": std(a29),
                "macro_recall_mean": mean(m29),
                "n_seed": len(a29),
                "aligned": True,
                "seeds": list(keep_seeds),
                "per_seed": a29,
                "per_seed_macro_recall": m29,
                "mean_30seed": mean([accs[s] for s in ALL_SEEDS]),
                "std_30seed": std([accs[s] for s in ALL_SEEDS]),
                "macro_recall_mean_30seed": mean([mrs[s] for s in ALL_SEEDS]),
            }}

    out = {
        "metric": "acc",
        "split": "val",
        "precision": "server_fp32",
        "note": ("サーバ FP32 の validation。**配備側 (summary_deploy_val.json) と"
                 "シード集合を揃えた版**。dinov2_l / dinov3_l は配備側に seed 42 が無い"
                 "ため seeds 43-71 の 29 本で再集計した。mnv4 / effb0 / resnet50 / "
                 "vit_small は両側とも seed 42-71 の 30 本で整合しているので "
                 "summary_val.json の値をそのまま複製している"),
        "aligned_to": os.path.relpath(os.path.join(R, "summary_deploy_val.json"), ROOT),
        "source": os.path.relpath(args.src, ROOT),
        "per_seed_source": os.path.relpath(PERSEED_DIR, ROOT),
        "alignment_reason": ("ViT-L (dinov2_l / dinov3_l) の配備精度は 1 シードあたり 126 "
                             "パートの分割ビルド・推論チェーンで測っており、seed 42 は"
                             "測っていない (seeds 43-71 の 29 本のみ)。サーバ側を 30 本の"
                             "ままにすると『演算精度の影響』と『seed 42 を含むか否かの影響』が"
                             "交絡し、両者を分離できない。母数を 29 本へ揃えることで"
                             "演算精度の影響だけを見られるようにする"),
        "seeds": list(ALL_SEEDS),
        "seeds_dropped_for_alignment": [DROP_SEED],
        "seeds_by_model": {**{m: list(ALL_SEEDS) for m in COPY_MODELS},
                           **{m: list(keep_seeds) for m in ALIGN_MODELS}},
        "models_copied_verbatim": list(COPY_MODELS),
        "models_realigned": list(ALIGN_MODELS),
        "regression_check": {
            "description": ("npz から seeds 42-71 の 30 本で集計した値が summary_val.json の "
                            "mean と一致することを ViT-L 全 28 構成で確認 (データ経路の回帰検査)"),
            "n_config_checked": len(raw),
            "tolerance": TOL,
            "max_abs_diff_mean": worst[1],
            "max_abs_diff_mean_config": worst[0],
            "max_abs_diff_macro_recall": worst_mr[1],
            "max_abs_diff_std": worst_std[1],
            "passed": True,
        },
        "resolutions": RES,
        "models": models,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")

    # ------------------------------------------------------------------ 表示
    print("\n=== 29 シードへ揃えた ViT-L の val 平均 (括弧は 30 シード版との差 pt) ===")
    print("%-10s %s" % ("model", "  ".join("%7d" % r for r in RES)))
    for m in ALIGN_MODELS:
        row29 = ["%7.4f" % models[m][str(r)]["B"]["mean"] for r in RES]
        rowd = ["%+7.3f" % (100 * (models[m][str(r)]["B"]["mean"]
                                   - sv["models"][m][str(r)]["B"]["mean"])) for r in RES]
        print("%-10s %s" % (m + " 29", "  ".join(row29)))
        print("%-10s %s" % ("  Δpt", "  ".join(rowd)))
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
