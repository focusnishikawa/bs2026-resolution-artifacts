#!/usr/bin/env python3
"""Phase 3: Orin の FP16 精度検証用に、前処理済み入力を raw binary で書き出す.

評価 (eval_sweep.py) と完全に同じ前処理を通すことが重要。ここがずれると
「FP16 で精度が落ちた」のか「入力が違った」のか区別できなくなる。

出力:
  inputs_r<N>.bin   float32 NCHW を 1882 枚ぶん連結 (1 枚 = 3*N*N)
  labels.npy        正解ラベル
  manifest_r<N>.json  枚数・寸法・sha256 の一部

usage:
    python3 train/prep_inputs_orin.py --resolutions 16 112 224 \
        --out_dir /home1/gfsi/ufsi0002/bs2026-resolution-edge/inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sweep import load_base_crops, make_tensor  # noqa: E402
from train_res import resolve_input_res  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/work/gfsi/ufsi0002/bs2026-raptor/data_root")
    p.add_argument("--splits_dir", default="/work/gfsi/ufsi0002/bs2026-raptor/data/splits")
    p.add_argument("--split", default="test")
    p.add_argument("--resolutions", nargs="*", type=int, default=[16, 112, 224])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", choices=["A", "B"], default="B")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crops, labels, rel_paths = load_base_crops(Path(args.splits_dir) / f"{args.split}.csv",
                                               Path(args.data_root))
    np.save(out_dir / "labels.npy", labels)
    print(f"[load] {len(crops)} crops / labels saved", flush=True)

    for res in args.resolutions:
        # CNN と patch16 ViT は input_res == res。dinov2_l だけ patch14 へ丸まるが、
        # 精度検証は代表モデル (CNN + ViT-S) で行うので res をそのまま使う。
        input_res = res if args.mode == "B" else 224
        t = make_tensor(crops, res, input_res)          # (N, 3, input_res, input_res) float32
        arr = t.numpy().astype(np.float32)
        path = out_dir / f"inputs_r{res}.bin"
        arr.tofile(path)
        h = hashlib.sha256(open(path, "rb").read(1 << 20)).hexdigest()[:16]
        meta = {"res": res, "input_res": input_res, "mode": args.mode,
                "n": int(arr.shape[0]), "shape": list(arr.shape),
                "bytes": path.stat().st_size, "sha256_head1MB": h,
                "dtype": "float32", "layout": "NCHW"}
        json.dump(meta, open(out_dir / f"manifest_r{res}.json", "w"), indent=2, ensure_ascii=False)
        print(f"[ok ] r{res:<4d} -> {path.name}  {arr.shape}  {path.stat().st_size/1e6:.1f} MB", flush=True)
        del t, arr


if __name__ == "__main__":
    main()
