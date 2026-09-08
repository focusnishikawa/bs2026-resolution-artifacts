#!/usr/bin/env python3
"""表 4 (FP16 の安全性) を全数推論 CSV から再集計する.

背景: 論文の Orin 値を TensorRT 10.3 へ全面差し替えする方針 (ユーザー指示 2026-08-31) で
全数推論をやり直したが、`fp16_accuracy.json` を作るスクリプトがリポジトリに無く
(`analyze_all.py` は読むだけ)、旧環境の値が取り残されていた。ここを埋める。

⭐ 参照は **サーバ (PyTorch FP32) の argmax** である。`collect_vitl_edge.py:296-303` の
   ViT-L と同じ定義にそろえてあるので、図 4 で CNN と ViT-L を同じ軸に並べられる。
   (Orin 上の FP32 エンジンとの比較ではない。両者は別物で、ViT-S/16 N=112 では
    サーバ基準 90.60% に対し Orin FP32 基準 90.54% と一致率が異なる)

   - fp32_acc          : サーバ FP32 の正解率      = (srv == labels)
   - fp16_acc          : Orin エンジンの正解率     = (edge == labels)
   - argmax_agreement  : サーバとの argmax 一致率  = (edge == srv)
   - n_mismatch        : 一致しなかった枚数

⚠️ 欠損を黙って飛ばさない。読めなかった構成は `missing` に出し、stdout にも警告する
   (壊れたエンジンや測り損ねを「無かったこと」にしないため)。

usage:
  python3 train/fp16_accuracy_trt10.py --preds <CSV ディレクトリ> --out <出力 json>
  python3 train/fp16_accuracy_trt10.py --preds results/orin/preds_orin --src legacy \
      --check results/orin/results/fp16_accuracy.json      # 旧値の再現テスト
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# ⛔ 2026-09-02: 既定を base から **s42** へ変更した。
#    Orin の CNN エンジンも models_s42 から書き出した ONNX で作られており (ONNX は 8/17 04:11
#    生成 = s42 preds 8/16 22:16 の後)、base を参照すると seed 差が半精度化の影響に混入する。
#    ⭐ 決め手は **FP32 同士の照合** — Orin の FP32 エンジン (vit_small_r112) との一致率は
#    base 90.54% (不一致 178 枚) に対し **s42 99.95% (不一致 1 枚 = 数値誤差)**。
#    FP16 の劣化が混ざらない比較なので、一致率で参照を選んでも誤らない。
#    ⚠️ 旧 fp16_accuracy_trt10.json は base 参照で作られていたので誤り。
DEF_SRV = os.path.join(ROOT, "results", "T1_condB_s42", "preds")

MODELS = ["mnv4", "effb0", "resnet50", "vit_small"]
# ⭐ 2026-09-02: 入力 bin を全 14 水準そろえ、全数推論も 56 構成で完了したので
#    ここも 14 水準へ広げた (それまでは inputs_r{16,112,224}.bin しか無く 3 水準だった)。
#    ⚠️ **これは「配備するエンジンそのものの task accuracy」を出すためである。**
#    サーバ FP32 の精度で代用すると、FP16 で崩れるモデル (ViT-S/16) の配備精度を
#    過大評価する。実測では ViT-S/16 は N=112 で 0.8932 -> 0.8427、N=224 で 0.9346 -> 0.8417。
#    --res で絞り込める (旧集計の再現には --res 16 112 224)。
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]

ENV = {
    "legacy": "JetPack 5.1.2 / TensorRT 8.5.2 / 15W",
    "trt10": "JetPack 6.2 (L4T R36.4.3) / TensorRT 10.3.0 / MAXN_SUPER",
}


def read_argmax(path):
    """全数推論 CSV の argmax 列を読む. 行数が合わなければ None を返す."""
    if not os.path.exists(path):
        return None, "CSV が無い"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "空の CSV"
    if "argmax" not in rows[0]:
        return None, "argmax 列が無い"
    return np.array([int(r["argmax"]) for r in rows]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="全数推論 CSV のディレクトリ")
    ap.add_argument("--srv", default=DEF_SRV, help="サーバ参照 npz のディレクトリ")
    ap.add_argument("--src", default="trt10", choices=sorted(ENV), help="測定環境のラベル")
    ap.add_argument("--out", default=None, help="出力 json (既定は stdout のみ)")
    ap.add_argument("--check", default=None, help="この json と突き合わせて回帰確認する")
    ap.add_argument("--res", nargs="*", type=int, default=None,
                    help="対象解像度 (既定は全 14 水準。旧集計の再現には --res 16 112 224)")
    args = ap.parse_args()
    global RES
    if args.res:
        RES = args.res

    if not os.path.isdir(args.srv):
        sys.exit("[abort] サーバ参照が無い: %s\n"
                 "  HPC の results/T1_condB/preds/*.npz を持ってくること" % args.srv)

    recs, missing = [], []
    for m in MODELS:
        for r in RES:
            tag = "%s_r%d" % (m, r)
            npz = os.path.join(args.srv, tag + ".npz")
            if not os.path.exists(npz):
                missing.append({"tag": tag, "why": "サーバ参照 npz が無い"})
                continue
            ref = np.load(npz)
            labels, srv = ref["labels"], ref["probs"].argmax(1)

            edge, why = read_argmax(os.path.join(args.preds, tag + ".csv"))
            if edge is None:
                missing.append({"tag": tag, "why": why})
                continue
            if len(edge) != len(labels):
                missing.append({"tag": tag,
                                "why": "行数不一致 (CSV %d / 参照 %d)" % (len(edge), len(labels))})
                continue

            rec = {
                "tag": tag, "model": m, "res": r,
                "fp32_acc": float((srv == labels).mean()),
                "fp16_acc": float((edge == labels).mean()),
                "argmax_agreement": float((edge == srv).mean()),
                "n_mismatch": int((edge != srv).sum()),
                "n": int(len(labels)),
            }
            # 相異なる出力の本数。FP16 の飽和で壊れたエンジンは定数を返すが、計算が
            # 消えるぶん latency では速く見えるので、ここでも数えておく。
            rec["distinct_argmax"] = int(len(np.unique(edge)))
            # Orin 上の FP32 エンジンがあれば、装置内の FP16 対 FP32 も出す (対照)
            dev, _ = read_argmax(os.path.join(args.preds, tag + "_FP32.csv"))
            if dev is not None and len(dev) == len(labels):
                rec["device_fp32_acc"] = float((dev == labels).mean())
                rec["fp16_vs_device_fp32"] = float((edge == dev).mean())
            recs.append(rec)

    print("=== 表 4: FP16 の精度劣化 (参照 = サーバ FP32 argmax, n=1882) ===")
    print("  出典: %s" % ENV[args.src])
    print("%-18s %10s %10s %12s %8s" % ("config", "FP32", "FP16", "argmax一致", "mismatch"))
    for r in sorted(recs, key=lambda x: (x["model"], x["res"])):
        print("%-18s %10.4f %10.4f %11.2f%% %8d"
              % (r["tag"], r["fp32_acc"], r["fp16_acc"], r["argmax_agreement"] * 100,
                 r["n_mismatch"]))
    if missing:
        print("  ⚠️ 欠損 %d 件: %s" % (len(missing),
                                     ", ".join("%s (%s)" % (x["tag"], x["why"]) for x in missing)))
    else:
        print("  被覆 %d/%d 構成" % (len(recs), len(MODELS) * len(RES)))

    ng = 0
    if args.check:
        old = {x["tag"]: x for x in json.load(open(args.check))}
        keys = ["fp32_acc", "fp16_acc", "argmax_agreement", "n_mismatch", "n"]
        print("\n=== 回帰確認: %s ===" % args.check)
        for r in sorted(recs, key=lambda x: (x["model"], x["res"])):
            o = old.get(r["tag"])
            if o is None:
                print("  %-18s 旧に無い" % r["tag"]); ng += 1; continue
            bad = [k for k in keys if o[k] != r[k]]
            print("  %-18s %s" % (r["tag"], "一致" if not bad else "不一致: %s" % bad))
            ng += len(bad) > 0
        print("  => %s (%d/%d 構成一致)" % ("PASS" if ng == 0 else "FAIL",
                                            len(recs) - ng, len(recs)))

    if args.out:
        payload = recs if args.src == "legacy" else recs
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
            f.write("\n")
        meta = os.path.splitext(args.out)[0] + "_meta.json"
        with open(meta, "w") as f:
            json.dump({"source": args.src, "measurement_env": ENV[args.src],
                       "reference": "server PyTorch FP32 argmax (results/T1_condB/preds/*.npz)",
                       "preds_dir": os.path.abspath(args.preds),
                       "n_configs": len(recs), "missing": missing,
                       "complete": not missing}, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("\n書き出し: %s\n          %s" % (args.out, meta))

    return 1 if (args.check and ng) else 0


if __name__ == "__main__":
    sys.exit(main())
