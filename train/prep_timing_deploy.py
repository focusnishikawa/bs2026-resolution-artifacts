#!/usr/bin/env python3
"""配備パイプラインの前処理レイテンシを Orin CPU 上で実測する.

`prep_timing_res.py` は評価用データセット (個体ごとに切り出された小さな JPEG) を
1 枚ずつ開く経路を測っており、そこで得た「デコード 3.4 ms + リサイズ 2.3 ms は
解像度に依存しない」という値は**オフライン評価の値**である。

配備時の経路はこれと異なる:

    [フレームあたり 1 回] 4K フレームをデコード (検出器が既に実施済み)
    [個体あたり]         メモリ上の 4K 配列から bbox を切り出し -> N x N へリサイズ -> 正規化

個体ごとの JPEG デコードは発生しない。本スクリプトは次の 3 つを測り、
両経路を同一機械で突き合わせられるようにする。

    A. 4K フレームのデコード (H.264 動画 / JPEG) = フレームあたりの共有コスト
    B. 個体あたりの前処理 (メモリ上の 4K 配列から切り出し -> N -> 正規化)
    C. 対照: `prep_timing_res.py` と同じオフライン経路

Orin には torch を入れていないので numpy + PIL + cv2 だけで完結させる。

usage (fgpu0):
    python3 prep_timing_deploy.py --clip clip_4k.mp4 --frame frame_4k.jpg \
        --n 200 --warmup 20 --out results/prep_timing_deploy.json
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BASE = 224
RESOLUTIONS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
# 実運用で検出器が返す bbox の一辺 [px]. 224 は論文のオフライン経路と揃えるための対照
BBOX_SIZES = [64, 112, 224]


def summarize(vals):
    v = sorted(vals)
    return {"mean": round(st.mean(v) * 1000, 4),
            "median": round(st.median(v) * 1000, 4),
            "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))] * 1000, 4),
            "n": len(v)}


def normalize(im_arr):
    a = im_arr.astype(np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1).copy()


# --- A. 4K フレームのデコード -------------------------------------------------
def measure_decode_video(path, warmup):
    """H.264 4K 動画のフレームデコード. 検出器がフレームごとに 1 回払うコスト."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("動画を開けない: %s" % path)
    ts, shape = [], None
    i = 0
    while True:
        t0 = time.perf_counter()
        ok, fr = cap.read()
        dt = time.perf_counter() - t0
        if not ok:
            break
        if shape is None:
            shape = fr.shape
        if i >= warmup:
            ts.append(dt)
        i += 1
    cap.release()
    return summarize(ts), shape


def measure_decode_jpeg(path, n, warmup):
    """4K JPEG のデコード (静止画経路の対照)."""
    ts_cv, ts_pil = [], []
    for i in range(n + warmup):
        t0 = time.perf_counter(); a = cv2.imread(path); dt = time.perf_counter() - t0
        if i >= warmup:
            ts_cv.append(dt)
        t0 = time.perf_counter(); im = Image.open(path).convert("RGB"); _ = np.asarray(im); dt = time.perf_counter() - t0
        if i >= warmup:
            ts_pil.append(dt)
    return {"cv2": summarize(ts_cv), "pil": summarize(ts_pil), "shape": list(a.shape)}


# --- B. 個体あたりの前処理 (配備経路) -----------------------------------------
def measure_per_bird(frame, res_list, bbox_sizes, n, warmup):
    """メモリ上の 4K 配列から bbox を切り出し -> N x N -> 正規化.

    切り出し位置はフレーム内を巡回させ、キャッシュに過度に有利にならないようにする。
    """
    H, W = frame.shape[:2]
    out = {}
    for B in bbox_sizes:
        out[str(B)] = {}
        for res in res_list:
            acc = {"crop": [], "resize_cv": [], "resize_pil": [], "norm": []}
            for i in range(n + warmup):
                # 巡回する切り出し位置 (決定的・乱数を使わない)
                x = (i * 137) % max(1, W - B)
                y = (i * 89) % max(1, H - B)

                t0 = time.perf_counter()
                crop = frame[y:y + B, x:x + B]
                crop = np.ascontiguousarray(crop)
                t_crop = time.perf_counter() - t0

                t0 = time.perf_counter()
                r_cv = crop if res == B else cv2.resize(crop, (res, res), interpolation=cv2.INTER_CUBIC)
                t_cv = time.perf_counter() - t0

                t0 = time.perf_counter()
                if res == B:
                    r_pil = crop
                else:
                    r_pil = np.asarray(Image.fromarray(crop).resize((res, res), Image.BICUBIC))
                t_pil = time.perf_counter() - t0

                t0 = time.perf_counter()
                _ = normalize(r_cv)
                t_norm = time.perf_counter() - t0

                if i >= warmup:
                    acc["crop"].append(t_crop)
                    acc["resize_cv"].append(t_cv)
                    acc["resize_pil"].append(t_pil)
                    acc["norm"].append(t_norm)
            rec = {k: summarize(v) for k, v in acc.items()}
            rec["total_cv"] = round(rec["crop"]["mean"] + rec["resize_cv"]["mean"] + rec["norm"]["mean"], 4)
            rec["total_pil"] = round(rec["crop"]["mean"] + rec["resize_pil"]["mean"] + rec["norm"]["mean"], 4)
            out[str(B)][str(res)] = rec
            print("  bbox=%3d res=%3d  crop %5.3f + resize(cv %5.3f / pil %5.3f) + norm %5.3f"
                  "  => total cv %6.3f ms / pil %6.3f ms"
                  % (B, res, rec["crop"]["mean"], rec["resize_cv"]["mean"], rec["resize_pil"]["mean"],
                     rec["norm"]["mean"], rec["total_cv"], rec["total_pil"]), flush=True)
    return out


# --- C. 対照: オフライン経路 (prep_timing_res.py と同一) -----------------------
def measure_offline(files, res_list, n, warmup):
    out = {}
    for res in res_list:
        acc = {"decode": [], "resize256": [], "crop224": [], "degrade": [], "norm": []}
        cnt = 0
        for i in range(n + warmup):
            f = files[i % len(files)]
            try:
                t0 = time.perf_counter(); im = Image.open(f).convert("RGB"); t_d = time.perf_counter() - t0
                t0 = time.perf_counter()
                w, h = im.size; s = 256 / min(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
                t_r = time.perf_counter() - t0
                t0 = time.perf_counter()
                W2, H2 = im.size; l = (W2 - BASE) // 2; u = (H2 - BASE) // 2
                im = im.crop((l, u, l + BASE, u + BASE))
                t_c = time.perf_counter() - t0
                t0 = time.perf_counter()
                if res != BASE:
                    im = im.resize((res, res), Image.BICUBIC)
                t_g = time.perf_counter() - t0
                t0 = time.perf_counter(); _ = normalize(np.asarray(im)); t_n = time.perf_counter() - t0
            except Exception:
                continue
            if i >= warmup:
                acc["decode"].append(t_d); acc["resize256"].append(t_r)
                acc["crop224"].append(t_c); acc["degrade"].append(t_g); acc["norm"].append(t_n)
                cnt += 1
        rec = {k: summarize(v) for k, v in acc.items()}
        rec["total"] = round(sum(rec[k]["mean"] for k in acc), 4)
        out[str(res)] = rec
        print("  [offline] res=%3d  decode %6.3f + resize256 %6.3f + crop %5.3f + degrade %5.3f + norm %5.3f = %6.3f ms"
              % (res, rec["decode"]["mean"], rec["resize256"]["mean"], rec["crop224"]["mean"],
                 rec["degrade"]["mean"], rec["norm"]["mean"], rec["total"]), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True, help="4K H.264 動画 (3840x1920)")
    p.add_argument("--frame", required=True, help="4K JPEG フレーム")
    p.add_argument("--offline_src", default=None, help="オフライン対照に使う画像ディレクトリ")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default="results/prep_timing_deploy.json")
    args = p.parse_args()

    out = {"n": args.n, "warmup": args.warmup,
           "note": "配備経路 (フレーム 1 回デコード + 個体ごと切り出し) と "
                   "オフライン経路 (個体ごとに JPEG を開く) を同一機械で比較する"}

    print("=== A. 4K フレームのデコード (フレームあたり 1 回) ===", flush=True)
    vid, shape = measure_decode_video(args.clip, args.warmup)
    print("  H.264 %s: mean %.3f ms / median %.3f ms (n=%d)"
          % (shape, vid["mean"], vid["median"], vid["n"]), flush=True)
    jpg = measure_decode_jpeg(args.frame, min(args.n, 50), args.warmup)
    print("  JPEG %s: cv2 mean %.3f ms / PIL mean %.3f ms"
          % (jpg["shape"], jpg["cv2"]["mean"], jpg["pil"]["mean"]), flush=True)
    out["frame_decode"] = {"h264": vid, "h264_shape": list(shape), "jpeg": jpg}

    print("", flush=True)
    print("=== B. 個体あたりの前処理 (メモリ上の 4K 配列から) ===", flush=True)
    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit("フレームを読めない: %s" % args.frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("  フレーム形状 %s" % (frame.shape,), flush=True)
    out["per_bird"] = measure_per_bird(frame, RESOLUTIONS, BBOX_SIZES, args.n, args.warmup)

    if args.offline_src:
        print("", flush=True)
        print("=== C. 対照: オフライン経路 (prep_timing_res.py と同一) ===", flush=True)
        import glob
        files = sorted(glob.glob(str(Path(args.offline_src) / "**" / "*.*"), recursive=True))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            out["offline"] = measure_offline(files, RESOLUTIONS, args.n, args.warmup)
        else:
            print("  画像が見つからないので対照は省略", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("", flush=True)
    print("[saved] %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
