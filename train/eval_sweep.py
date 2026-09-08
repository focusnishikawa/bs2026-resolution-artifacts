#!/usr/bin/env python3
"""条件 A (劣化耐性) の解像度スイープ評価.

224 で学習したモデルを固定し、評価時のみ入力を N へ落として 224 へ復元する。
条件 B (解像度別ネイティブ学習) の再評価にも --mode B で使える。

効率化: 224 基準クロップを一度だけメモリに載せ、全解像度で使い回すため画像 I/O は 1 回で済む。
CI や検定は生値から後段で計算できるよう、softmax 確率と正解ラベルを npz に必ず保存する。

usage:
    python3 train/eval_sweep.py --mode A --models_dir models --output_dir results/T1_condA \\
        --data_root <...> --splits_dir <...>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_res import (  # noqa: E402
    MODEL_SPECS, RESOLUTIONS, BASE_RES, MEAN, STD,
    build_model, resolve_input_res, DegradeResolution,
)

TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)


def load_base_crops(csv_path: Path, data_root: Path):
    """全評価画像を Resize(256) -> CenterCrop(224) した PIL 画像として読み込む。"""
    df = pd.read_csv(csv_path)
    base_tf = transforms.Compose([
        transforms.Resize(int(BASE_RES * 1.143)),
        transforms.CenterCrop(BASE_RES),
    ])
    crops = []
    t0 = time.time()
    for i, rel in enumerate(df["rel_path"]):
        crops.append(base_tf(Image.open(Path(data_root) / rel).convert("RGB")))
        if (i + 1) % 500 == 0:
            print(f"  [load] {i + 1}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    labels = df["species_idx"].values.astype(np.int64)
    return crops, labels, df["rel_path"].tolist()


def make_tensor(crops, res: int, input_res: int, workers: int = 16) -> torch.Tensor:
    """224 基準クロップを解像度 N へ落とし、必要なら入力寸法へ戻してテンソル化する。

    PIL の resize は GIL を解放するのでスレッド並列が効く。単一スレッドだと 1,882 枚で
    4 分近くかかり、ここが評価全体の律速になる。
    """
    degrade = DegradeResolution(res, restore_to=input_res if input_res != res else None)

    def one(im):
        return NORMALIZE(TO_TENSOR(degrade(im)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        tensors = list(ex.map(one, crops))
    return torch.stack(tensors)


@torch.no_grad()
def infer(model, imgs: torch.Tensor, device, batch_size: int, use_amp: bool) -> np.ndarray:
    model.eval()
    outs = []
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else None
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i:i + batch_size].to(device, non_blocking=True)
        if amp_ctx is not None:
            with amp_ctx:
                logits = model(chunk)
        else:
            logits = model(chunk)
        outs.append(logits.float().cpu().numpy())
    logits = np.concatenate(outs, 0)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def metrics_from(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    preds = probs.argmax(1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    per_class_recall = [float(cm[i, i] / max(1, cm[i, :].sum())) for i in range(num_classes)]
    per_class_prec = [float(cm[i, i] / max(1, cm[:, i].sum())) for i in range(num_classes)]
    top5 = None
    if num_classes > 5:
        order = np.argsort(-probs, axis=1)[:, :5]
        top5 = float(np.mean([labels[i] in order[i] for i in range(len(labels))]))
    return {
        "acc": float((preds == labels).mean()),
        "macro_recall": float(np.mean(per_class_recall)),
        "macro_precision": float(np.mean(per_class_prec)),
        "per_class_recall": per_class_recall,
        "top5_acc": top5,
        "cm": cm.tolist(),
        "n": int(len(labels)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["A", "B"], default="A")
    p.add_argument("--models_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--splits_dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    p.add_argument("--resolutions", nargs="*", type=int, default=RESOLUTIONS)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--summary_name", default="summary.json",
                   help="複数プロセスで分割実行する際に上書き競合を避けるため分ける")
    p.add_argument("--torch_threads", type=int, default=8)
    args = p.parse_args()

    # 複数プロセスを同一ノードで並走させるため、torch の CPU スレッドを絞って
    # 過剰サブスクリプション (48 コアを各プロセスが取り合う状態) を避ける
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (args.output_dir / "preds").mkdir(parents=True, exist_ok=True)

    print(f"[load] base crops from {args.split}.csv", flush=True)
    crops, labels, rel_paths = load_base_crops(args.splits_dir / f"{args.split}.csv", args.data_root)
    print(f"[load] {len(crops)} crops", flush=True)

    summary = {"mode": args.mode, "split": args.split, "n": len(crops),
               "resolutions": args.resolutions, "results": {}}

    for model_id in args.models:
        summary["results"][model_id] = {}

    # 解像度を外側にして、同じ入力寸法のテンソルをモデル間で使い回す。
    # 条件 A は全モデルの入力が 224 なので、解像度あたり 1 回の変換で済む。
    for res in args.resolutions:
        tensor_cache: dict[int, torch.Tensor] = {}
        for model_id in args.models:
            spec = MODEL_SPECS[model_id]
            tag = f"{model_id}_r{res}"
            out_json = args.output_dir / f"{tag}.json"
            if args.skip_existing and out_json.exists():
                summary["results"][model_id][str(res)] = json.load(open(out_json))
                print(f"[skip] {tag}", flush=True)
                continue

            # 条件 A は 224 学習済みの重みを使い回す。条件 B は解像度ごとの重みを読む。
            ckpt_dir = args.models_dir / (f"{model_id}_r{BASE_RES}" if args.mode == "A" else tag)
            ckpt_path = ckpt_dir / "best.pt"
            if not ckpt_path.exists():
                print(f"[miss] {tag}: {ckpt_path} が無い -> スキップ", flush=True)
                continue

            input_res = resolve_input_res(model_id, res, args.mode)
            t0 = time.time()
            if input_res not in tensor_cache:
                tensor_cache[input_res] = make_tensor(crops, res, input_res)
            imgs = tensor_cache[input_res]

            model, _ = build_model(model_id, args.num_classes, input_res, None, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)

            bs = args.batch_size if spec["kind"] != "vit_l" else min(args.batch_size, 64)
            probs = infer(model, imgs, device, bs, use_amp=spec["amp"])
            m = metrics_from(probs, labels, args.num_classes)
            m.update({"model": model_id, "res": res, "input_res": input_res, "mode": args.mode,
                      "ckpt": str(ckpt_path), "eval_sec": round(time.time() - t0, 1)})

            json.dump(m, open(out_json, "w"), indent=2, ensure_ascii=False)
            np.savez_compressed(args.output_dir / "preds" / f"{tag}.npz",
                                probs=probs.astype(np.float32), labels=labels,
                                rel_paths=np.array(rel_paths))
            summary["results"][model_id][str(res)] = m
            print(f"[{args.mode}] {model_id:10s} res={res:3d} input={input_res:3d} "
                  f"acc={m['acc']:.4f} mR={m['macro_recall']:.4f} ({m['eval_sec']}s)", flush=True)

            del model, probs
            torch.cuda.empty_cache()
        del tensor_cache

    json.dump(summary, open(args.output_dir / args.summary_name, "w"), indent=2, ensure_ascii=False)
    print(f"\n[done] {args.output_dir}/{args.summary_name}", flush=True)


if __name__ == "__main__":
    main()
