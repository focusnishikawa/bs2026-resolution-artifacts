#!/usr/bin/env python3
"""DINOv3-L の ONNX から **If ノードだけ**を畳み、他のノードには一切触れない.

なぜ必要か
----------
既存の `fold_if.py` は onnxruntime の `ORT_ENABLE_BASIC` に通してグラフ全体を最適化する。
If は確かに消えるが (2 -> 0、ノード 3,600 -> 2,714)、**同時にグラフ全体が書き換わる**。
DINOv2-L では「分割前に ORT 最適化を通すか否か」だけで latency が 1.7-2.5 倍変わることが
実測されており (TensorRT の融合が効かなくなるため)、DINOv3-L の 1 段が DINOv2-L より
1.4 倍重い (8.21 対 5.82 ms/段) のも同じ機序ではないかという疑いが残っている。

本スクリプトはその疑いを検証するために、**If の分岐選択だけ**を行う。
ノードの融合・除去・並べ替えは一切しないので、ORT 最適化の影響を切り分けられる。

手順
----
  1. 条件テンソルの値を ORT で評価する。入力 shape が [1,3,N,N] に固定されているので
     Greater(Sub(Size(x), c1), c2) は一意に決まる。
     607 MB の本体を読み込まずに済むよう、条件までの部分グラフを抽出してから評価する。
  2. 各 If について真になる分岐のサブグラフを親グラフへインライン展開する。
     サブグラフ内のテンソル名にはプレフィックスを付けて親との衝突を避ける。
  3. If ノードを削除し、どこからも参照されなくなったノードだけを落とす。

usage:
    python3 fold_if_minimal.py <src.onnx> <dst.onnx>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


def count_ops(path):
    m = onnx.load(str(path), load_external_data=False)
    c = {}
    for n in m.graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return c, len(m.graph.node)


def _np_dtype(elem_type):
    """ONNX の elem_type から numpy dtype を得る (FP16 モデルでは入力も fp16)"""
    try:
        return onnx.helper.tensor_dtype_to_np_dtype(elem_type)
    except AttributeError:          # 古い onnx 向けの後方互換
        from onnx import mapping
        return mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type]


def eval_conditions(src, cond_names, input_name, input_shape, elem_type):
    """条件テンソルの値を得る。条件までの部分グラフだけを抽出して実行する"""
    import onnxruntime as ort
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "cond_sub.onnx")
    onnx.utils.extract_model(str(src), tmp, [input_name], list(cond_names))
    so = ort.SessionOptions()
    # ⚠️ ここで最適化を掛けると本末転倒なので必ず切る
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
    dummy = np.zeros(input_shape, dtype=_np_dtype(elem_type))
    outs = sess.run(list(cond_names), {input_name: dummy})
    os.remove(tmp)
    return {n: bool(np.asarray(v).reshape(-1)[0]) for n, v in zip(cond_names, outs)}


def inline_branch(if_node, branch, taken):
    """選ばれた分岐のノード列を、親グラフに置ける形へ変換して返す"""
    pre = (if_node.name or "If").strip("/").replace("/", "_") + "_%s__" % taken
    # サブグラフが生成するテンソルだけを改名する (外部スコープ参照はそのまま)
    produced = {o for n in branch.node for o in n.output}
    ren = {t: pre + t.lstrip("/") for t in produced}

    nodes = []
    for n in branch.node:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
        nn.name = pre + (n.name or n.op_type).lstrip("/")
        for i, t in enumerate(nn.input):
            if t in ren:
                nn.input[i] = ren[t]
        for i, t in enumerate(nn.output):
            nn.output[i] = ren[t]
        nodes.append(nn)

    # 分岐の出力を If の出力名へ繋ぐ (Identity 1 個だけ増える)
    for bo, io in zip(branch.output, if_node.output):
        src_name = ren.get(bo.name, bo.name)
        nodes.append(helper.make_node("Identity", [src_name], [io],
                                      name=pre + "to_" + io.lstrip("/").replace("/", "_")))
    inits = list(branch.initializer)
    for t in inits:
        if t.name in ren:
            t.name = ren[t.name]
    return nodes, inits


def prune_unused(graph):
    """どこからも参照されないノードを落とす (If が消えて宙に浮いた条件計算など)"""
    keep_names = {o.name for o in graph.output}
    changed = True
    removed = 0
    while changed:
        changed = False
        used = set(keep_names)
        for n in graph.node:
            used.update(n.input)
        drop = [i for i, n in enumerate(graph.node)
                if n.output and not any(o in used for o in n.output)]
        for i in reversed(drop):
            del graph.node[i]
            removed += 1
            changed = True
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()

    before, nb = count_ops(args.src)
    print("[before] nodes=%d  If=%d" % (nb, before.get("If", 0)), flush=True)
    if not before.get("If"):
        print("[skip] If が無いのでコピーするだけ")
        args.dst.write_bytes(args.src.read_bytes())
        return 0

    m = onnx.load(str(args.src))
    g = m.graph
    inp = g.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print("[input ] %s %s" % (inp.name, shape), flush=True)

    ifs = [(i, n) for i, n in enumerate(g.node) if n.op_type == "If"]
    conds = [n.input[0] for _, n in ifs]
    print("[cond  ] %s" % conds, flush=True)
    val = eval_conditions(args.src, conds, inp.name, shape, inp.type.tensor_type.elem_type)
    for c in conds:
        print("         %s = %s" % (c, val[c]), flush=True)

    # 後ろの If から処理して添字のずれを避ける
    add_inits = []
    for idx, node in reversed(ifs):
        taken = "then" if val[node.input[0]] else "else"
        br = None
        for a in node.attribute:
            if a.name == ("then_branch" if val[node.input[0]] else "else_branch"):
                br = a.g
        assert br is not None, "分岐が見つからない: %s" % node.name
        nodes, inits = inline_branch(node, br, taken)
        del g.node[idx]
        for k, nn in enumerate(nodes):
            g.node.insert(idx + k, nn)
        add_inits.extend(inits)
        print("[fold  ] %s -> %s 分岐 (%d ノードを展開)" % (node.name, taken, len(nodes)), flush=True)
    g.initializer.extend(add_inits)

    removed = prune_unused(g)
    print("[prune ] 未参照ノードを %d 個削除" % removed, flush=True)

    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, str(args.dst), save_as_external_data=False)

    after, na = count_ops(args.dst)
    print("[after ] nodes=%d  If=%d" % (na, after.get("If", 0)), flush=True)
    print("[size  ] %.1f MB -> %.1f MB" % (args.src.stat().st_size / 1e6, args.dst.stat().st_size / 1e6))
    custom = [n.op_type for n in onnx.load(str(args.dst), load_external_data=False).graph.node
              if n.domain not in ("", "ai.onnx")]
    print("[opset ] 非標準ドメイン: %s" % (sorted(set(custom)) if custom else "なし (TRT がパース可能)"))
    print("[RESULT] %s" % ("If は完全に消えた" if after.get("If", 0) == 0 else "If が残っている"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
