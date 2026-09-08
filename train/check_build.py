#!/usr/bin/env python3
"""Phase 0 スモーク: 全 (モデル x 解像度) 構成でモデル構築と forward が通るかを一括検証する.

学習に入る前に、以下を潰しておくのが目的:
  - N=16 で patch16 ViT がトークン 1 個になっても forward が通るか
  - DINOv2-L の pos_embed 補間が全解像度で成立するか
  - 事前学習重みがオフライン環境で実際にロードできるか (missing key の確認)
  - 条件 A (224 復元) と条件 B (ネイティブ) の両方で入力寸法が意図どおりか

出力: results/build_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_res import MODEL_SPECS, RESOLUTIONS, build_model, resolve_input_res, build_transforms  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    p.add_argument("--modes", nargs="*", default=["B", "A"])
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    n_ok = n_ng = 0

    for model_id in args.models:
        for mode in args.modes:
            # 条件 A は入力が常に 224 なのでモデル構築は 1 回で足りる
            res_list = RESOLUTIONS if mode == "B" else [RESOLUTIONS[-1]]
            for res in res_list:
                input_res = resolve_input_res(model_id, res, mode)
                rec = {"model": model_id, "mode": mode, "res": res, "input_res": input_res}
                t0 = time.time()
                try:
                    model, info = build_model(model_id, args.num_classes, input_res,
                                              args.pretrain_dir, pretrained=True)
                    model.eval().to(device)
                    x = torch.randn(args.batch, 3, input_res, input_res, device=device)
                    with torch.no_grad():
                        y = model(x)
                    ok_shape = tuple(y.shape) == (args.batch, args.num_classes)
                    rec.update({
                        "ok": bool(ok_shape),
                        "out_shape": list(y.shape),
                        "params_M": round(sum(q.numel() for q in model.parameters()) / 1e6, 3),
                        "other_missing": len(info.get("other_missing", [])),
                        "unexpected": len(info.get("unexpected", [])),
                        "pos_embed_resampled_to": info.get("pos_embed_resampled_to"),
                        "sec": round(time.time() - t0, 2),
                    })
                    patch = MODEL_SPECS[model_id]["patch"]
                    if patch:
                        rec["tokens"] = (input_res // patch) ** 2
                    if not ok_shape:
                        rec["error"] = f"想定外の出力形状: {tuple(y.shape)}"
                    # 重みロードの取りこぼしは致命的なので明示的に落とす
                    if rec["other_missing"] > 0:
                        rec["ok"] = False
                        rec["error"] = f"head 以外の重み欠損 {rec['other_missing']} 件"
                    del model, x, y
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc().splitlines()[-3:],
                                "sec": round(time.time() - t0, 2)})
                # 前処理側も同時に検証する (入力寸法が transform の出力と一致するか)
                try:
                    _, val_tf = build_transforms(res, input_res)
                    from PIL import Image
                    dummy = Image.new("RGB", (640, 480), (128, 128, 128))
                    t = val_tf(dummy)
                    rec["tf_shape"] = list(t.shape)
                    if tuple(t.shape) != (3, input_res, input_res):
                        rec["ok"] = False
                        rec["error"] = f"transform 出力 {tuple(t.shape)} が入力寸法 {input_res} と不一致"
                except Exception as e:
                    rec["ok"] = False
                    rec["error"] = f"transform: {type(e).__name__}: {e}"

                rows.append(rec)
                n_ok += int(rec.get("ok", False))
                n_ng += int(not rec.get("ok", False))
                mark = "ok " if rec.get("ok") else "NG "
                print(f"[{mark}] {model_id:10s} mode={mode} res={res:3d} input={input_res:3d} "
                      f"tokens={rec.get('tokens', '-')} params={rec.get('params_M', '-')}M "
                      f"{rec.get('error', '')}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_ok": n_ok, "n_ng": n_ng, "rows": rows}, open(args.output, "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[summary] ok={n_ok} ng={n_ng} -> {args.output}", flush=True)
    return 1 if n_ng else 0


if __name__ == "__main__":
    sys.exit(main())
