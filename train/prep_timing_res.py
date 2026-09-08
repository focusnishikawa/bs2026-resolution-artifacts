#!/usr/bin/env python3
"""Phase 3: Orin CPU 上の前処理レイテンシを解像度別に実測する.

学習・評価と同一の前処理を段階ごとに計る:
    decode -> resize(256) -> center crop(224) -> [N へ縮小] -> [条件 A なら 224 へ復元]
    -> to tensor -> normalize

GPU 推論が 1-7 ms と速いため、前処理が支配的になるかどうかが配備上の焦点になる。
解像度を下げても decode と resize(256) のコストは変わらない点に注意 (ここが効くはず)。

Orin には torch を入れていないので numpy + PIL だけで完結させる。

usage (fgpu0):
    python3 prep_timing_res.py --src <画像ディレクトリ> --n 200 --warmup 20
"""
import argparse
import glob
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BASE = 224
RESOLUTIONS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]


def preprocess(path, res, restore_to=None):
    t = {}
    t0 = time.perf_counter(); im = Image.open(path).convert("RGB"); t["decode"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    w, h = im.size; s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    t["resize256"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    W, H = im.size; l = (W - BASE) // 2; u = (H - BASE) // 2
    im = im.crop((l, u, l + BASE, u + BASE))
    t["crop224"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if res != BASE:
        im = im.resize((res, res), Image.BICUBIC)
    if restore_to is not None and restore_to != res:
        im = im.resize((restore_to, restore_to), Image.BICUBIC)
    t["degrade"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    a = np.asarray(im, np.float32) / 255.0
    a = (a - MEAN) / STD
    a = a.transpose(2, 0, 1).copy()
    t["tensor_norm"] = time.perf_counter() - t0
    return t


def summarize(vals):
    v = sorted(vals)
    return {"mean": round(st.mean(v) * 1000, 4), "median": round(st.median(v) * 1000, 4),
            "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))] * 1000, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="画像ディレクトリ or ファイルリスト")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--mode", choices=["A", "B"], default="B")
    p.add_argument("--out", default="results/prep_timing.json")
    p.add_argument("--resolutions", nargs="*", type=int, default=RESOLUTIONS)
    args = p.parse_args()

    src = Path(args.src)
    if src.is_dir():
        files = sorted(glob.glob(str(src / "**" / "*.*"), recursive=True))
    else:
        files = [l.strip() for l in open(src) if l.strip()]
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))][:args.n + args.warmup]
    if len(files) < args.warmup + 10:
        raise SystemExit("画像が足りない: %d 枚" % len(files))
    print("[prep] %d 枚 (warmup %d) / mode=%s" % (len(files), args.warmup, args.mode), flush=True)

    out = {"mode": args.mode, "n": len(files) - args.warmup, "base": BASE, "by_res": {}}
    stages = ["decode", "resize256", "crop224", "degrade", "tensor_norm"]
    for res in args.resolutions:
        restore = BASE if args.mode == "A" else None
        acc = {k: [] for k in stages}
        tot = []
        for i, f in enumerate(files):
            try:
                t = preprocess(f, res, restore)
            except Exception:
                continue
            if i >= args.warmup:
                for k in stages:
                    acc[k].append(t[k])
                tot.append(sum(t.values()))
        rec = {k: summarize(acc[k]) for k in stages}
        rec["total"] = summarize(tot)
        out["by_res"][str(res)] = rec
        print("  res=%3d  total mean=%7.3f ms  (decode %6.3f / resize256 %6.3f / crop %5.3f / degrade %5.3f / norm %5.3f)"
              % (res, rec["total"]["mean"], rec["decode"]["mean"], rec["resize256"]["mean"],
                 rec["crop224"]["mean"], rec["degrade"]["mean"], rec["tensor_norm"]["mean"]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("[saved] %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
