#!/usr/bin/env python3
"""p3 エンジンを単独で叩くための中間テンソル (p0->p2 の出力) を作る.

probe_p3.py により、4 段チェーンの誤りは p3 に局在し、しかも p3 の出力が入力に
依存していない (1,882 枚で相異なる出力が dinov2_l 3 個 / dinov3_l 1 個) ことが分かった。
残る分岐は 2 つ:

  (a) p3 エンジン自体が入力を無視している (TRT ビルドの問題)
  (b) 段間の受け渡しで p3 に正しい入力が届いていない (チェーン実行側の問題)

これを切り分けるには **p3 を 1 段だけのチェーンとして実行**し、正しい中間テンソルを
ファイルから直接与えればよい。ここではその入力バイナリと、ORT による参照 logits を作る。

入力の値そのものが効いているかを見るため、通常の中間テンソルに加えて
0.5 倍に縮めた変種も書き出す (p3 が入力を読んでいれば logits は必ず変わる)。

usage:
    python3 gen_mid_input.py [tag] [n]
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

EDGE = Path("/home1/gfsi/ufsi0002/bs2026-resolution-edge")
RES = {"dinov2_l_r32": 28, "dinov3_l_r32": 32}


def sess(p):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def run_parts(tag, parts, h):
    for p in parts:
        s = sess(EDGE / ("onnx_split/%s_%s.onnx" % (tag, p)))
        name = s.get_inputs()[0].name
        out = [s.run(None, {name: h[i:i + 1]})[0] for i in range(len(h))]
        h = np.concatenate(out, 0)
    return h


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "dinov2_l_r32"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    res = RES[tag]

    x = np.fromfile(EDGE / ("inputs/real_%s.fp16.bin" % tag), dtype=np.float16)
    x = x.reshape(-1, 3, res, res)[:n]
    print("[input] %s" % (x.shape,))

    mid = run_parts(tag, ["p0", "p1", "p2"], x)
    print("[mid] shape=%s dtype=%s 値域 [%.3f, %.3f]"
          % (mid.shape, mid.dtype, mid.min(), mid.max()))

    # p3 は分割 ONNX 側の宣言に合わせて fp16 で受け渡す
    mid16 = mid.astype(np.float16)
    half16 = (mid.astype(np.float32) * 0.5).astype(np.float16)

    for name, arr in [("mid", mid16), ("half", half16)]:
        path = EDGE / ("inputs/%s_p012_%s.fp16.bin" % (tag, name))
        arr.tofile(path)
        ref = run_parts(tag, ["p3"], arr).reshape(len(arr), -1).astype(np.float32)
        np.save(EDGE / ("inputs/%s_p3ref_%s.npy" % (tag, name)), ref)
        print("  %-5s -> %s (%d B) / 参照 logits argmax 先頭8=%s"
              % (name, path.name, path.stat().st_size, ref.argmax(1)[:8].tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
