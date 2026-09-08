#!/usr/bin/env python3
"""Jetson Orin Nano 上で 4K H.264 のデコードを CPU と NVDEC で比較する.

`prep_timing_deploy.py` で 4K フレームの CPU デコードが 28.9 ms/フレームと分かり、
10 fps 時の実効上限 31.3 ms の 92% を占めることが判明した。残る 2.4 ms で検出処理が
成立するとは考えにくいので、ハードウェアデコーダ (NVDEC) を使った場合を測る。

3 経路を同一素材で比較する。

  (a) CPU        cv2.VideoCapture 既定 (FFMPEG)  -> BGR ホストメモリ
  (b) NVDEC->BGR nvv4l2decoder ! nvvidconv ! videoconvert ! appsink -> BGR ホストメモリ
  (c) NVDEC のみ nvv4l2decoder ! fakesink (NVMM 内で完結。別スクリプトで測る)

⚠️ (b) を測るのが要点である。NVDEC の出力は NV12 のデバイスメモリなので、既存の
検出器が期待する BGR のホスト配列にするには色変換とコピーが要る。これを込みで測らないと
NVDEC が不当に有利に見える。(c) は検出器を NVMM/CUDA 直結へ書き換えた場合の下限にあたる。

あわせて CPU と NVDEC のデコード結果の画素差も出す。デコーダを替えても検出の入力が
変わらないこと (あるいはどれだけ変わるか) を確認するため。

usage (fgpu0):
    python3 nvdec_bench.py --video clip.mp4 --n 200 --warmup 10 --out nvdec_bench.json
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import cv2
import numpy as np

NVDEC_PIPELINE = (
    "filesrc location={path} ! qtdemux ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=false sync=false max-buffers=2"
)


def summarize(vals):
    if not vals:
        return None
    v = sorted(vals)
    return {"mean": round(st.mean(v) * 1000, 4),
            "median": round(st.median(v) * 1000, 4),
            "sd": round(st.pstdev(v) * 1000, 4) if len(v) > 1 else 0.0,
            "min": round(v[0] * 1000, 4),
            "max": round(v[-1] * 1000, 4),
            "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))] * 1000, 4),
            "n": len(v)}


def bench(cap, n, warmup, label):
    """1 フレーム読むのにかかる時間を測る (デコード + ホストへの取り出し)."""
    ts = []
    shape = None
    i = 0
    t_start = time.perf_counter()
    while i < n + warmup:
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
    wall = time.perf_counter() - t_start
    s = summarize(ts)
    if s is None:
        print("  [%s] フレームを読めなかった" % label, flush=True)
        return None, shape, i
    print("  [%-12s] mean %7.3f ms / median %7.3f ms / sd %6.3f (n=%d)  実効 %.1f fps"
          % (label, s["mean"], s["median"], s["sd"], s["n"], i / wall if wall else 0), flush=True)
    return s, shape, i


def open_cpu(path):
    cap = cv2.VideoCapture(str(path))
    return cap


def open_nvdec(path):
    return cv2.VideoCapture(NVDEC_PIPELINE.format(path=path), cv2.CAP_GSTREAMER)


def compare_pixels(path, k):
    """CPU と NVDEC のデコード結果を先頭 k フレームで突き合わせる."""
    a, b = open_cpu(path), open_nvdec(path)
    if not (a.isOpened() and b.isOpened()):
        a.release(); b.release()
        return None
    diffs = []
    for _ in range(k):
        oka, fa = a.read()
        okb, fb = b.read()
        if not (oka and okb):
            break
        if fa.shape != fb.shape:
            diffs.append({"error": "shape 不一致 %s vs %s" % (fa.shape, fb.shape)})
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        diffs.append({"mean_abs": float(d.mean()), "max_abs": int(d.max()),
                      "frac_diff": float((d > 0).mean()),
                      "frac_gt2": float((d > 2).mean())})
    a.release(); b.release()
    ok = [d for d in diffs if "error" not in d]
    if not ok:
        return {"frames": 0, "detail": diffs}
    return {"frames": len(ok),
            "mean_abs": round(float(np.mean([d["mean_abs"] for d in ok])), 4),
            "max_abs": int(max(d["max_abs"] for d in ok)),
            "frac_diff": round(float(np.mean([d["frac_diff"] for d in ok])), 4),
            "frac_gt2": round(float(np.mean([d["frac_gt2"] for d in ok])), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--cmp_frames", type=int, default=10)
    p.add_argument("--out", default="nvdec_bench.json")
    args = p.parse_args()

    path = Path(args.video)
    if not path.exists():
        raise SystemExit("動画が無い: %s" % path)

    out = {"video": str(path), "n": args.n, "warmup": args.warmup,
           "note": "4K H.264 のデコードを CPU と NVDEC で比較する。"
                   "NVDEC 側は BGR ホストメモリまで含めた値である"}

    print("=== (a) CPU デコード (cv2 既定バックエンド) ===", flush=True)
    cap = open_cpu(path)
    if not cap.isOpened():
        raise SystemExit("CPU 経路で開けない")
    out["backend_cpu"] = cap.getBackendName()
    print("  backend = %s" % out["backend_cpu"], flush=True)
    s, shape, cnt = bench(cap, args.n, args.warmup, "CPU")
    cap.release()
    out["cpu"] = s
    out["shape"] = list(shape) if shape else None
    out["frames_read_cpu"] = cnt

    print("", flush=True)
    print("=== (b) NVDEC -> BGR (ホストメモリまで) ===", flush=True)
    cap = open_nvdec(path)
    if not cap.isOpened():
        print("  NVDEC パイプラインを開けなかった", flush=True)
        out["nvdec_bgr"] = None
    else:
        s, shape_n, cnt_n = bench(cap, args.n, args.warmup, "NVDEC->BGR")
        cap.release()
        out["nvdec_bgr"] = s
        out["frames_read_nvdec"] = cnt_n
        out["shape_nvdec"] = list(shape_n) if shape_n else None

    if out.get("cpu") and out.get("nvdec_bgr"):
        r = out["cpu"]["mean"] / out["nvdec_bgr"]["mean"]
        out["speedup_cpu_over_nvdec_bgr"] = round(r, 3)
        print("", flush=True)
        print("  CPU / NVDEC->BGR = %.2f 倍" % r, flush=True)

    print("", flush=True)
    print("=== (d) デコード結果の画素差 (CPU 対 NVDEC) ===", flush=True)
    cmp_ = compare_pixels(path, args.cmp_frames)
    out["pixel_diff"] = cmp_
    if cmp_ and cmp_.get("frames"):
        print("  %d フレーム: 平均絶対差 %.4f / 最大 %d / 差のある画素 %.2f%% (差>2 は %.2f%%)"
              % (cmp_["frames"], cmp_["mean_abs"], cmp_["max_abs"],
                 100 * cmp_["frac_diff"], 100 * cmp_["frac_gt2"]), flush=True)
    else:
        print("  比較できなかった: %s" % cmp_, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("", flush=True)
    print("[saved] %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
