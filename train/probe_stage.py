#!/usr/bin/env python3
"""分割チェーンがどの段で壊れているかを特定する.

ONNX (onnxruntime) では分割前後が完全ビット一致したのに、Orin の TRT では
4 分割チェーンの精度が 0.12 まで落ちる。dtype・バイト数・フォーマット (kLINEAR) は
すべて一致しているので、段の出力そのものを ORT の同じ段の出力と突き合わせる。
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def read_csv(path):
    rows = []
    with open(path) as f:
        f.readline()
        for line in f:
            v = line.strip().split(",")
            rows.append([float(x) for x in v[1:-1]])   # idx と argmax を除く
    return np.asarray(rows, dtype=np.float64)


def sess(p):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def main():
    edge = Path("/home1/gfsi/ufsi0002/bs2026-resolution-edge")
    n = 4
    x = np.fromfile(edge / "inputs/real_dinov2_l_r32.fp16.bin", dtype=np.float16)
    x = x.reshape(-1, 3, 28, 28)[:n]
    print("[input] %s" % (x.shape,))

    FP16_MAX = 65504.0
    for tag, parts, csv in [("p0", ["p0"], "probe_dinov2_p0.csv"),
                            ("p0+p1", ["p0", "p1"], "probe_dinov2_p01.csv"),
                            ("p0+p1+p2", ["p0", "p1", "p2"], "probe_dinov2_p012.csv")]:
        h = x
        for p in parts:
            s = sess(edge / ("onnx_split/dinov2_l_r32_%s.onnx" % p))
            out = []
            for i in range(len(h)):
                out.append(s.run(None, {s.get_inputs()[0].name: h[i:i+1]})[0])
            h = np.concatenate(out, 0)
        ref = h.reshape(n, -1).astype(np.float64)
        got = read_csv(edge / ("preds_orin/%s" % csv))
        print("\n=== %s ===" % tag)
        print("  ORT   shape=%s  値域 [%.3f, %.3f]" % (ref.shape, ref.min(), ref.max()))
        print("  Orin  shape=%s  値域 [%.3f, %.3f]" % (got.shape, got.min(), got.max()))
        # 段の境界テンソルが fp16 のダイナミックレンジに収まっているか
        # (収まらなければ TRT が段間で inf/NaN を撒く。ORT 側の値で判定する)
        over = int((np.abs(ref) > FP16_MAX).sum())
        print("  [fp16 レンジ] 最大絶対値=%.1f (fp16 上限 %.0f)  超過要素=%d 個"
              % (np.abs(ref).max(), FP16_MAX, over))
        if ref.shape != got.shape:
            print("  [NG] 形が違う")
            continue
        d = np.abs(ref - got)
        print("  最大絶対差=%.6g  平均絶対差=%.6g" % (d.max(), d.mean()))
        # 並びが違うだけなら、ソートした値は一致するはず
        rs, gs = np.sort(ref[0]), np.sort(got[0])
        print("  [並び検査] 値をソートして比較した最大差=%.6g" % np.abs(rs - gs).max())
        print("  ORT  先頭8: %s" % np.array2string(ref[0][:8], precision=3))
        print("  Orin 先頭8: %s" % np.array2string(got[0][:8], precision=3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
