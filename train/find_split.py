#!/usr/bin/env python3
"""ViT-L の ONNX を transformer ブロック境界で分割するための切断点を洗い出す.

経路 A の下ごしらえ。Orin の OOM は Transformer 全体が単一の MYELIN ForeignNode へ融合され、
その一括コンパイルがメモリを要求することが原因なので、ブロック境界でグラフを切って
別々のエンジンにすればビルド時のピークが下がる。

切断点として使えるのは「そこを通る値が 1 本のテンソルだけ」になる箇所 (グラフのくびれ)。
各ブロック出口のテンソルについて、それより前のノードの出力が後ろでどれだけ参照されるかを
数え、残存参照が 1 本だけの位置を候補として出す。
"""
import argparse
from collections import defaultdict

import onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--nparts", type=int, default=4)
    args = p.parse_args()

    m = onnx.load(args.src, load_external_data=False)
    g = m.graph
    init = {t.name for t in g.initializer}
    n = len(g.node)
    print("nodes=%d initializers=%d input=%s" % (n, len(init), [i.name for i in g.input]))

    # 各テンソルが最後に消費されるノード位置。
    # Constant の出力は分割時に各パートへ複製すればよく、パート間の依存にならないので数えない。
    produced_at = {}
    n_const = 0
    for i, node in enumerate(g.node):
        if node.op_type == "Constant":
            n_const += 1
            continue
        for o in node.output:
            produced_at[o] = i
    print("Constant ノード %d 個は切断の依存から除外" % n_const)
    last_use = defaultdict(lambda: -1)
    for i, node in enumerate(g.node):
        for x in node.input:
            if x in produced_at:
                last_use[x] = i

    # 位置 i と i+1 の間を横切る「生きているテンソル」の本数
    cuts = []
    for i in range(n - 1):
        live = [t for t, pi in produced_at.items() if pi <= i and last_use[t] > i]
        cuts.append((i, live))

    narrow = [(i, live) for i, live in cuts if len(live) == 1]
    print("\nくびれ (横断テンソル 1 本) は %d 箇所" % len(narrow))

    # ブロック名から何番目のブロックかを推定して表示する
    def blk(i):
        nm = g.node[i].name
        for part in nm.split("/"):
            if part.isdigit():
                return part
        return "?"

    step = max(1, len(narrow) // (args.nparts * 3))
    print("%-8s %-6s %-46s %s" % ("node#", "block", "横断テンソル", "直後のノード"))
    for i, live in narrow[::step][:40]:
        print("%-8d %-6s %-46s %s" % (i, blk(i), live[0][:46], g.node[i + 1].op_type))

    # nparts 等分に最も近いくびれを選ぶ
    print("\n=== %d 分割の推奨切断点 ===" % args.nparts)
    chosen = []
    for k in range(1, args.nparts):
        target = n * k // args.nparts
        best = min(narrow, key=lambda x: abs(x[0] - target))
        chosen.append(best)
        print("  part%d/%d 目標 node#%-5d -> 実際 node#%-5d (block %s) テンソル=%s"
              % (k, args.nparts, target, best[0], blk(best[0]), best[1][0]))
    if not chosen:
        print("  くびれが見つからず等分できない")


if __name__ == "__main__":
    main()
