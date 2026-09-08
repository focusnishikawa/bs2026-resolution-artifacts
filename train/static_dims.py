#!/usr/bin/env python3
"""分割 ONNX に残った symbolic な batch 次元を静的な 1 に固定する.

split_onnx.py は分割の境界テンソルをそのまま graph input/output に昇格させるため、
元モデルの batch 次元 (dim_param) が symbolic のまま残る。この状態で trtexec に
渡すと

    [W] Dynamic dimensions required for input: ... Automatically overriding shape to: 1x5x1024

が出る。p1/p2 はそれでも正しく動くが、**p3 のエンジンだけ入力を完全に無視した定数を返す**
(実機で確認: 中間テンソルを 0.5 倍しても出力が 1 ビットも変わらない)。
ONNX 自体は ORT で入力依存に動くので、TRT のビルド側の問題である。

そこで symbolic 次元を最初から静的に潰した ONNX を作り、自動オーバーライドを介さずに
ビルドできるようにする。

usage:
    python3 static_dims.py <tag> [parts...]     # 既定は p0 p1 p2 p3
"""
import sys
from pathlib import Path

import onnx

EDGE = Path("/home1/gfsi/ufsi0002/bs2026-resolution-edge")


def fix(vi, batch=1):
    """value_info の symbolic 次元 (dim_param) を batch に置き換える"""
    changed = []
    dims = vi.type.tensor_type.shape.dim
    for i, d in enumerate(dims):
        if d.HasField("dim_param") or (not d.HasField("dim_value")):
            changed.append((i, d.dim_param))
            d.ClearField("dim_param")
            d.dim_value = batch
        elif d.dim_value == 0:
            changed.append((i, "0"))
            d.dim_value = batch
    return changed


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "dinov2_l_r32"
    parts = sys.argv[2:] or ["p0", "p1", "p2", "p3"]

    for p in parts:
        src = EDGE / ("onnx_split/%s_%s.onnx" % (tag, p))
        dst = EDGE / ("onnx_split/%s_%s_static.onnx" % (tag, p))
        m = onnx.load(str(src))
        n_in = [(vi.name, fix(vi)) for vi in m.graph.input]
        n_out = [(vi.name, fix(vi)) for vi in m.graph.output]
        # 中間の value_info にも symbolic が残ると shape 推論が通らないことがあるので消す
        del m.graph.value_info[:]
        onnx.save(m, str(dst))
        print("[%s] %s -> %s" % (p, src.name, dst.name))
        for name, ch in n_in:
            print("   IN  %-45s 固定した次元=%s" % (name, ch if ch else "なし"))
        for name, ch in n_out:
            print("   OUT %-45s 固定した次元=%s" % (name, ch if ch else "なし"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
