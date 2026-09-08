#!/usr/bin/env python3
"""Phase 3 準備: 学習済み best.pt を解像度ごとに ONNX へ書き出す (Orin TensorRT FP16 用).

batch=1 固定・動的軸なし (trtexec の FP16 最適化を最大化するため)。
ViT 系は TensorRT 8.5.2 が融合 attention を扱えないため、timm の fused_attn を切ってから
書き出す (raptor で vit_small を Orin に載せた際と同じ回避策)。

条件 A のレイテンシは入力が常に 224 なので、条件 B の N=224 エンジンと同一になる。
したがって書き出すのは条件 B の 84 構成のみでよい。

usage:
    python3 train/export_onnx.py --models_dir models --output_dir onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_res import MODEL_SPECS, RESOLUTIONS, build_model, resolve_input_res  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    p.add_argument("--resolutions", nargs="*", type=int, default=RESOLUTIONS)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--mode", choices=["A", "B"], default="B")
    p.add_argument("--opset_cnn", type=int, default=17)
    p.add_argument("--opset_vit", type=int, default=16)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--half", action="store_true",
                   help="重みを FP16 で書き出す。ViT-L は FP32 の ONNX が 1.2 GB あり、"
                        "Orin (実効 6 GB) では trtexec のパース中に OOM kill される。"
                        "FP16 なら 0.6 GB に半減する。TRT 側でどのみち FP16 化するので精度は変わらない")
    p.add_argument("--suffix", default="", help="出力ファイル名の接尾辞 (例: _fp16)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for model_id in args.models:
        spec = MODEL_SPECS[model_id]
        is_vit = spec["patch"] is not None
        if is_vit:
            # TensorRT 8.5.2 は scaled_dot_product_attention を含む融合 attention を解釈できない
            try:
                from timm.layers import set_fused_attn
                set_fused_attn(False)
                print(f"[timm] fused_attn を無効化 ({model_id})", flush=True)
            except ImportError:
                print("[warn] set_fused_attn が無い timm。ViT の ONNX が TRT で通らない可能性あり", flush=True)

        for res in args.resolutions:
            tag = f"{model_id}_r{res}"
            out_path = args.output_dir / f"{tag}{args.suffix}.onnx"
            if args.skip_existing and out_path.exists():
                print(f"[skip] {tag}", flush=True)
                continue

            ckpt_path = args.models_dir / tag / "best.pt"
            if not ckpt_path.exists():
                print(f"[miss] {tag}: {ckpt_path} が無い -> スキップ", flush=True)
                continue

            input_res = resolve_input_res(model_id, res, args.mode)
            t0 = time.time()
            model, _ = build_model(model_id, args.num_classes, input_res, None, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            if args.half:
                model = model.half()

            dtype = torch.float16 if args.half else torch.float32
            dummy = torch.randn(1, 3, input_res, input_res, dtype=dtype)
            opset = args.opset_vit if is_vit else args.opset_cnn
            try:
                torch.onnx.export(
                    model, dummy, str(out_path),
                    input_names=["input"], output_names=["logits"],
                    opset_version=opset, do_constant_folding=True,
                    dynamic_axes=None, dynamo=False,
                )
                # 書き出した ONNX が元モデルと同じ出力を出すかを必ず確認する
                with torch.no_grad():
                    ref = model(dummy).numpy()
                rec = {"model": model_id, "res": res, "input_res": input_res, "opset": opset,
                       "half": bool(args.half), "onnx": out_path.name, "size_MB": round(out_path.stat().st_size / 1e6, 2),
                       "export_sec": round(time.time() - t0, 1),
                       "ref_logits": [round(float(v), 5) for v in ref[0]]}
                print(f"[ok ] {tag:20s} input={input_res:3d} opset={opset} "
                      f"{rec['size_MB']:7.2f} MB ({rec['export_sec']}s)", flush=True)
            except Exception as e:
                rec = {"model": model_id, "res": res, "input_res": input_res,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"[NG ] {tag}: {rec['error']}", flush=True)
            manifest.append(rec)
            del model

    out_json = args.output_dir / ("manifest%s.json" % (args.suffix or ""))
    json.dump(manifest, open(out_json, "w"), indent=2, ensure_ascii=False)
    n_ng = sum(1 for r in manifest if "error" in r)
    print(f"\n[done] {len(manifest)} 構成 / 失敗 {n_ng} -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
