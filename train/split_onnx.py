#!/usr/bin/env python3
"""ViT-L の ONNX を transformer ブロック境界で N 分割する (経路 A).

Orin の OOM は Transformer 全体が単一 MYELIN ForeignNode に融合され、その一括コンパイルが
メモリを要求することが原因。ブロック境界で切って別エンジンにすればビルド時のピークが下がる。

切断点は find_split.py が出す「くびれ」(横断テンソル 1 本の位置) を使う。各ブロック出口が
すべてくびれになっているので、パート間で受け渡すテンソルは常に 1 本だけで済む。

usage:
    python3 split_onnx.py src.onnx out_dir --nparts 4
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import shape_inference


def const_closure(graph):
    """初期化子だけから計算できるテンソル (重みの Cast など) の集合を求める。

    ORT 最適化を通した ONNX には InsertedPrecisionFreeCast_* のような重み前処理ノードが
    混ざる。これらはデータの流れではないので切断の依存に数えてはいけない。
    Constant だけを除外すると重みキャストを本流と誤認し、切断点が壊れる。
    """
    const = {t.name for t in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            const.update(n.output)
    changed = True
    while changed:
        changed = False
        for n in graph.node:
            if n.output and n.input and all(x in const for x in n.input):
                for o in n.output:
                    if o not in const:
                        const.add(o)
                        changed = True
    return const


def find_cut_tensors(model, nparts):
    """ブロック境界のくびれから nparts 等分に近い切断テンソルを選ぶ。"""
    g = model.graph
    const = const_closure(g)
    produced_at = {}
    for i, node in enumerate(g.node):
        if node.op_type == "Constant":
            continue
        for o in node.output:
            if o in const:      # 重み由来のテンソルは本流ではない
                continue
            produced_at[o] = i
    last_use = defaultdict(lambda: -1)
    for i, node in enumerate(g.node):
        for x in node.input:
            if x in produced_at:
                last_use[x] = i

    n = len(g.node)
    narrow = []
    for i in range(n - 1):
        live = [t for t, pi in produced_at.items() if pi <= i and last_use[t] > i]
        if len(live) == 1:
            narrow.append((i, live[0]))

    cuts = []
    for k in range(1, nparts):
        target = n * k // nparts
        best = min(narrow, key=lambda x: abs(x[0] - target))
        if best[1] not in [c[1] for c in cuts]:
            cuts.append(best)
    return cuts, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--nparts", type=int, default=4)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.src.stem

    print("[load] %s (%.1f MB)" % (args.src, args.src.stat().st_size / 1e6), flush=True)
    model = onnx.load(str(args.src))
    # extract_model は中間テンソルの value_info を要求するので先に shape 推論を通す
    model = shape_inference.infer_shapes(model, strict_mode=False)
    inferred = args.out_dir / ("%s_inferred.onnx" % tag)
    onnx.save(model, str(inferred))

    cuts, n_nodes = find_cut_tensors(model, args.nparts)
    print("[cut ] nodes=%d 切断テンソル=%s" % (n_nodes, [c[1] for c in cuts]), flush=True)

    g_in = [i.name for i in model.graph.input]
    g_out = [o.name for o in model.graph.output]
    bounds = [g_in] + [[c[1]] for c in cuts] + [g_out]

    manifest = {"src": args.src.name, "nparts": args.nparts, "parts": []}
    for k in range(len(bounds) - 1):
        ins, outs = bounds[k], bounds[k + 1]
        dst = args.out_dir / ("%s_p%d.onnx" % (tag, k))
        print("[part%d] %s -> %s" % (k, ins, outs), flush=True)
        onnx.utils.extract_model(str(inferred), str(dst), ins, outs)
        sz = dst.stat().st_size
        sub = onnx.load(str(dst), load_external_data=False)
        manifest["parts"].append({
            "part": k, "onnx": dst.name, "size_MB": round(sz / 1e6, 2),
            "nodes": len(sub.graph.node), "inputs": ins, "outputs": outs,
        })
        print("        %.1f MB / %d nodes" % (sz / 1e6, len(sub.graph.node)), flush=True)

    inferred.unlink()
    mf = args.out_dir / ("%s_split.json" % tag)
    json.dump(manifest, open(mf, "w"), indent=2, ensure_ascii=False)
    print("[done] %d パート -> %s" % (args.nparts, mf), flush=True)
    print("        合計 %.1f MB (元 %.1f MB)"
          % (sum(p["size_MB"] for p in manifest["parts"]), args.src.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
