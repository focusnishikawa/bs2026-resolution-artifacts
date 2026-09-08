#!/usr/bin/env python3
"""分割チェーンの最終段 (p3) が壊れていることを数値で確定させる.

probe_stage.py により p0 / p0+p1 / p0+p1+p2 の中間テンソルはいずれも ORT と
fp16 誤差の範囲で一致することが分かった (相対誤差 0.7% 程度で一定)。
残るのは p3 だけなので、ここでは 3 つの logits を突き合わせて p3 エンジン自体の
妥当性を判定する:

  A: ORT p3 に **Orin が実際に p2 まで計算した中間テンソル** を入れた結果
     (= p3 の入力は実機と同一。p3 の計算だけを ORT に置き換えた対照)
  B: ORT だけで p0->p3 を通した結果 (純粋な参照値)
  C: Orin が 4 段チェーンで出した logits

C が A/B から大きく外れれば、入力が正しくても p3 エンジンの出力が誤り = p3 の
TRT ビルドが原因と確定する。A と B が一致することは、前段の fp16 誤差が最終段の
argmax を覆すほどではないことの確認にもなる。
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def read_csv(path, n=None):
    rows = []
    with open(path) as f:
        f.readline()
        for line in f:
            v = line.strip().split(",")
            rows.append([float(x) for x in v[1:-1]])   # idx と argmax を除く
            if n is not None and len(rows) >= n:
                break
    return np.asarray(rows, dtype=np.float64)


def sess(p):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def run_parts(edge, tag, parts, h):
    for p in parts:
        s = sess(edge / ("onnx_split/%s_%s.onnx" % (tag, p)))
        out = [s.run(None, {s.get_inputs()[0].name: h[i:i + 1]})[0] for i in range(len(h))]
        h = np.concatenate(out, 0)
    return h


def main():
    edge = Path("/home1/gfsi/ufsi0002/bs2026-resolution-edge")
    tag = "dinov2_l_r32"
    n = 4

    x = np.fromfile(edge / "inputs/real_dinov2_l_r32.fp16.bin", dtype=np.float16)
    x = x.reshape(-1, 3, 28, 28)[:n]

    # A: Orin の p2 出力 (実機の中間テンソル) を ORT の p3 に入れる
    mid = read_csv(edge / "preds_orin/probe_dinov2_p012.csv", n)
    mid = mid.reshape(n, 5, 1024).astype(np.float16)
    A = run_parts(edge, tag, ["p3"], mid).reshape(n, -1).astype(np.float64)

    # B: ORT だけで通した参照値
    B = run_parts(edge, tag, ["p0", "p1", "p2", "p3"], x).reshape(n, -1).astype(np.float64)

    # C: Orin の 4 段チェーンが出した logits
    C = read_csv(edge / ("preds_orin/%s_split4_FP16.csv" % tag), n)

    print("=== logits 比較 (%s, 先頭 %d 枚) ===" % (tag, n))
    for name, v in [("A ORT_p3(Orin中間)", A), ("B ORT 全段", B), ("C Orin 4段チェーン", C)]:
        print("\n[%s] shape=%s argmax=%s" % (name, v.shape, v.argmax(1).tolist()))
        for i in range(n):
            print("   img%d: %s" % (i, np.array2string(v[i], precision=3, suppress_small=False)))

    # D: p3 にゼロを入れた場合の出力。Orin の 4 段チェーンは 4 枚とも同じ logits を
    # 返しており「入力に依存していない」ので、p3 が前段の出力ではなく未初期化 (ゼロ)
    # のバッファを読んでいる可能性がある。一致すれば計算誤りではなく受け渡しの問題。
    zero = np.zeros((1, 5, 1024), dtype=np.float16)
    D = run_parts(edge, tag, ["p3"], zero).reshape(1, -1).astype(np.float64)
    print("\n[D ORT_p3(ゼロ入力)] argmax=%d" % D.argmax())
    print("   %s" % np.array2string(D[0], precision=3))
    print("   C との最大絶対差=%.4g" % np.abs(D[0] - C[0]).max())

    print("\n=== 差分 ===")
    for name, u, v in [("A vs B (前段 fp16 誤差の影響)", A, B),
                       ("C vs B (実機チェーン vs 参照)", C, B),
                       ("C vs A (p3 エンジンの妥当性)", C, A),
                       ("C vs D (ゼロ入力仮説)", C, np.repeat(D, n, 0))]:
        d = np.abs(u - v)
        same = int((u.argmax(1) == v.argmax(1)).sum())
        print("  %-32s 最大絶対差=%-10.4g 平均=%-10.4g argmax一致=%d/%d"
              % (name, d.max(), d.mean(), same, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
