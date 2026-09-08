#!/usr/bin/env python3
"""分割済みパートを 1 つずつ onnxruntime の基本最適化に通す.

DINOv3-L は N=32 の検証で、**パート単位で ORT 最適化を掛け直すと**
  (a) 定数を返していた p2/p3 が正しく動くようになり
  (b) 各段の latency がほぼ半減する (p0 7.27->3.67 / p1 12.99->6.10 /
      p2 13.63->6.63 / p3 13.68->6.69 ms)
ことが分かっている。モデル全体を一度に最適化した ONNX を分割しただけでは足りない。

usage:
    python3 ort_parts.py <split_dir> <tag> [nparts]
      例: ort_parts.py <edge>/onnx_split dinov3_l_r64_fp16_sim 4
"""
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


def main():
    split_dir = Path(sys.argv[1])
    tag = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    for k in range(n):
        src = split_dir / ("%s_p%d.onnx" % (tag, k))
        dst = split_dir / ("%s_p%d_ort.onnx" % (tag, k))
        if not src.exists():
            print("[miss] %s" % src)
            continue
        if dst.exists():
            print("[skip] %s (既存)" % dst.name)
            continue
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = str(dst)
        ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
        print("[ok] p%d: ノード %d -> %d (%.1f -> %.1f MB)"
              % (k, len(onnx.load(str(src)).graph.node), len(onnx.load(str(dst)).graph.node),
                 src.stat().st_size / 1e6, dst.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
