#!/usr/bin/env python3
"""dinov3_l の ONNX に残る If ノードを解析し、TRT がパースに失敗する原因を特定する.

TRT 8.5.2 のエラーは
    /If_OutputLayer: IIfConditionalOutputLayer inputs must have the same shape.
であり If 自体の非対応ではない。then/else 両分岐の出力 shape が食い違っていることが原因なので、
その食い違いの中身と、条件入力が定数に畳めるかどうかを見る。
"""
import sys

import onnx
from onnx import shape_inference

path = sys.argv[1]
m = onnx.load(path)
g = m.graph
print("ir=%s opset=%s producer=%s" % (m.ir_version, [o.version for o in m.opset_import], m.producer_name))
print("nodes=%d inputs=%s" % (len(g.node), [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]))

# 初期化子 (定数) の名前集合。条件入力がここにあれば静的に畳める
init = {t.name for t in g.initializer}
print("initializers=%d" % len(init))

for idx, n in enumerate(g.node):
    if n.op_type != "If":
        continue
    print("\n===== node[%d] %s (op=If) =====" % (idx, n.name))
    print("  inputs =%s" % list(n.input))
    print("  outputs=%s" % list(n.output))
    cond = n.input[0]
    print("  cond '%s' は initializer か: %s" % (cond, cond in init))
    # 条件を生成しているノードを遡る
    for p in g.node:
        if cond in p.output:
            print("  cond の生成元: op=%s name=%s inputs=%s" % (p.op_type, p.name, list(p.input)))
            for pp in g.node:
                if any(o in p.input for o in pp.output):
                    print("    その上流: op=%s name=%s inputs=%s" % (pp.op_type, pp.name, list(pp.input)))
    for attr in n.attribute:
        sub = attr.g
        print("  --- branch '%s': %d nodes ---" % (attr.name, len(sub.node)))
        for o in sub.output:
            dims = []
            for d in o.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
            print("      out %-24s shape=%s elem=%s" % (o.name, dims, o.type.tensor_type.elem_type))
        for sn in sub.node[:12]:
            print("      %-14s %-40s <- %s" % (sn.op_type, sn.name[:40], list(sn.input)[:3]))
        if len(sub.node) > 12:
            print("      ... 他 %d ノード" % (len(sub.node) - 12))
