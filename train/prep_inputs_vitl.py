#!/usr/bin/env python3
"""ViT-L の Orin FP16 検証用に、前処理済み入力を fp16 raw binary で書き出す.

prep_inputs_orin.py の ViT-L 版。既存版と違うのは 2 点だけ:
  - ViT-L の ONNX は重みごと fp16 で書き出してあるので入力も fp16 でなければならない
  - dinov2_l は patch14 のため入力寸法が res=32 -> 28 に丸まる。モデルと transform の
    両方に丸め後の値を渡さないと評価時と違う前処理になる (ここがずれると
    「FP16 で落ちた」のか「入力が違った」のか区別できなくなる)

評価 (eval_sweep.py) と完全に同じ前処理を通すため make_tensor をそのまま使う。

usage:
    python3 prep_inputs_vitl.py --models dinov2_l dinov3_l --res 32 --out_dir <dir>
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
    p.add_argument("--models", nargs="*", default=["dinov2_l", "dinov3_l"])
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crops, labels, rel_paths = load_base_crops(Path(args.splits_dir) / f"{args.split}.csv",
                                               Path(args.data_root))
    print(f"[load] {len(crops)} crops", flush=True)

    for model_id in args.models:
        input_res = resolve_input_res(model_id, args.res, "B")
        t = make_tensor(crops, args.res, input_res)      # (N, 3, input_res, input_res) float32
        arr = t.numpy().astype(np.float16)               # ViT-L の ONNX は fp16 入力
        path = out_dir / f"real_{model_id}_r{args.res}.fp16.bin"
        arr.tofile(path)
        h = hashlib.sha256(open(path, "rb").read(1 << 20)).hexdigest()[:16]
        meta = {"model": model_id, "res": args.res, "input_res": input_res, "mode": "B",
                "n": int(arr.shape[0]), "shape": list(arr.shape),
                "bytes": path.stat().st_size, "sha256_head1MB": h,
                "dtype": "float16", "layout": "NCHW",
                "bytes_per_image": int(arr[0].nbytes)}
        json.dump(meta, open(out_dir / f"manifest_real_{model_id}_r{args.res}.json", "w"),
                  indent=2, ensure_ascii=False)
        print(f"[ok ] {model_id:9s} input_res={input_res:3d} {arr.shape} "
              f"{arr[0].nbytes} B/枚 {path.stat().st_size/1e6:.1f} MB -> {path.name}", flush=True)
        del t, arr

    np.save(out_dir / "labels.npy", labels)
    print(f"[ok ] labels {labels.shape} -> labels.npy", flush=True)


if __name__ == "__main__":
    main()
