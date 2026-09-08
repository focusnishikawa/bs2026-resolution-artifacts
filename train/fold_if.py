#!/usr/bin/env python3
"""dinov3_l の ONNX から静的に決まる If を畳み落として TRT 8.5.2 がパースできる形にする.

TRT の失敗は `IIfConditionalOutputLayer inputs must have the same shape` であり、If 自体は
サポートされている。残っている If の条件は Greater(Sub(Size(x), c1), c2) で、入力 shape が
[1,3,N,N] に固定されている以上すべて静的に決まる。onnxruntime の基本最適化 (constant folding)
に通せば条件が定数化し、分岐がインライン展開されて If が消える。

ORT_ENABLE_BASIC に留めるのは、EXTENDED 以上だと com.microsoft ドメインの独自演算に
置き換わり TRT がパースできなくなるため。
"""
import argparse
import shutil
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


def count_ops(path):
    m = onnx.load(path, load_external_data=False)
    n = {}
    for node in m.graph.node:
        n[node.op_type] = n.get(node.op_type, 0) + 1
    return n, len(m.graph.node)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    args = p.parse_args()

    before, nb = count_ops(args.src)
    print("[before] nodes=%d  If=%d  Size=%d  Greater=%d  Less=%d"
          % (nb, before.get("If", 0), before.get("Size", 0),
             before.get("Greater", 0), before.get("Less", 0)), flush=True)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(args.dst)
    # 最適化済みモデルを書き出すのが目的なので実行はしない
    sess = ort.InferenceSession(str(args.src), so, providers=["CPUExecutionProvider"])
    del sess

    if not args.dst.exists():
        print("[NG] 最適化モデルが書き出されなかった", file=sys.stderr)
        return 1

    after, na = count_ops(args.dst)
    print("[after ] nodes=%d  If=%d  Size=%d  Greater=%d  Less=%d"
          % (na, after.get("If", 0), after.get("Size", 0),
             after.get("Greater", 0), after.get("Less", 0)), flush=True)
    print("[size  ] %.1f MB -> %.1f MB" % (args.src.stat().st_size / 1e6, args.dst.stat().st_size / 1e6))

    # ORT が独自ドメインの演算を混ぜていないか確認する (混ざると TRT がパースできない)
    m = onnx.load(str(args.dst), load_external_data=False)
    doms = sorted({o.domain for o in m.opset_import})
    print("[opset ] %s" % [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    custom = [n.op_type for n in m.graph.node if n.domain not in ("", "ai.onnx")]
    if custom:
        print("[warn  ] 非標準ドメインの演算が %d 個: %s" % (len(custom), sorted(set(custom))[:10]))
    else:
        print("[ok    ] 全ノードが標準 opset (TRT がパース可能な形)")

    if after.get("If", 0) == 0:
        print("[RESULT] If は完全に消えた")
    else:
        print("[RESULT] If が %d 個残っている" % after["If"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
