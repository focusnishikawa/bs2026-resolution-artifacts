#!/usr/bin/env python3
"""ViT-L の Orin FP16 実測とサーバ FP32 を突き合わせる.

結論 4「ViT は FP16 で壊れる」は vit_small でしか確かめていない (argmax 一致 85.92%、-9.78pt)。
ViT-L でも同じことが起きるならエッジ配備の可否そのものが変わるので、同じ土俵
(サーバ FP32 の probs vs Orin FP16 の logits、全 1882 枚) で比較する。

分割エンジンと単体エンジンの両方がある場合は、実機上でも両者が一致することを併せて確認する
(ONNX レベルでは完全ビット一致を確認済み)。
"""
import sys
from pathlib import Path

import numpy as np


def read_csv(path):
    """orin_infer_chain の出力 (idx,l0..lk,argmax) を読む。"""
    rows = []
    with open(path) as f:
        head = f.readline().strip().split(",")
        ncls = len(head) - 2
        for line in f:
            v = line.strip().split(",")
            if len(v) != ncls + 2:
                continue
            rows.append([float(x) for x in v[1:1 + ncls]])
    return np.asarray(rows, dtype=np.float64)


def main():
    edge = Path("/home1/gfsi/ufsi0002/bs2026-resolution-edge")
    work = Path("/work/gfsi/ufsi0002/bs2026-resolution")

    print("%-10s %-14s %8s %8s %9s %9s" % ("model", "engine", "サーバ", "Orin", "差(pt)", "argmax一致"))
    print("-" * 68)
    store = {}
    for model in ["dinov2_l", "dinov3_l"]:
        ref = np.load(work / ("results/T1_condB/preds/%s_r32.npz" % model))
        probs, labels = ref["probs"], ref["labels"]
        srv_pred = probs.argmax(1)
        srv_acc = float((srv_pred == labels).mean())

        # split4fix = 最終段 p3 を ORT 基本最適化経由でビルドし直したチェーン
        # (元の p3 エンジンは入力を無視して定数を返していた。詳細は FINDINGS.md §5)
        # split4opt  = 全パートをパート単位の ORT 最適化版でビルドし直したチェーン
        # split5fp32 = p3 をさらに 2 分割し、FP16 では壊れる後半だけ FP32 にした 5 段チェーン
        for kind in ["split4", "split4fix", "split4opt", "split5fp32", "single"]:
            csv = edge / ("preds_orin/%s_r32_%s_FP16.csv" % (model, kind))
            if not csv.exists() or csv.stat().st_size == 0:
                continue
            logits = read_csv(csv)
            if len(logits) != len(labels):
                print("%-10s %-14s 行数 %d != %d でスキップ" % (model, kind, len(logits), len(labels)))
                continue
            orin_pred = logits.argmax(1)
            orin_acc = float((orin_pred == labels).mean())
            agree = float((orin_pred == srv_pred).mean())
            store[(model, kind)] = orin_pred
            print("%-10s %-14s %8.4f %8.4f %+9.2f %8.2f%%"
                  % (model, kind, srv_acc, orin_acc, (orin_acc - srv_acc) * 100, agree * 100))

    # 実機上での分割 vs 単体 (ONNX では完全一致を確認済み。TRT でも一致するか)
    print()
    for model in ["dinov2_l", "dinov3_l"]:
        for kind in ["split4", "split4fix"]:
            a, b = store.get((model, kind)), store.get((model, "single"))
            if a is not None and b is not None:
                print("[実機] %s: %s vs 単体エンジンの argmax 一致 = %.2f%% (%d/%d)"
                      % (model, kind, (a == b).mean() * 100, int((a == b).sum()), len(a)))

    # 参考: vit_small の既知の値と並べる
    print("\n[参考] 既存の結論 4 (vit_small @ r224): サーバ 0.9346 -> Orin FP16 0.8374 "
          "(-9.78 pt) / argmax 一致 85.92%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
