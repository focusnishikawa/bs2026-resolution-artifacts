#!/usr/bin/env python3
"""Phase 2: 条件 B (解像度別ネイティブ学習) を GPU 4 基へ動的に割り当てて実行する.

実行時間はモデルと解像度で 2 桁違うため、静的に分配すると特定 GPU だけが残る。ここでは
  - 見積りコストの降順にジョブを並べ (LPT スケジューリング)
  - 空いた GPU が次のジョブを取る動的キュー
とすることで全体の終了時刻を縮める。

完了済み (test_metrics.json が存在) のジョブは既定でスキップするため、中断しても再実行で続きから進む。

学習上限について (v2 で変更):
  初版は CNN 80 epoch / ViT-L 30 epoch で回したが、84 構成中 24 構成が early stop に達せず
  上限で打ち切られ、mnv4_r144 などは best_epoch が 79/80 と明確に未収束だった。
  そのため上限と patience を引き上げた。**cosine スケジュールは total_epochs に依存するので、
  一部の構成だけ延長すると統制が揃わない。上限を変えたら全構成を学習し直すこと。**

seed について:
  seed ごとに models ディレクトリを分ける (models_s42 / models_s43 / ...)。
  こうすると評価側 (eval_sweep.py) は --models_dir を差し替えるだけで済む。

usage (hpciaiss1):
    nohup setsid python3 train/run_phase2.py --seed 42 --models_dir models_s42 \
        --progress_name p2_s42.json > logs/phase2_s42.log 2>&1 &

注意: 本スクリプトは apptainer を起動する側なので SIF の外 (ノードのシステム python) で動く。
そのシステム python は 3.6.8 と古く `from __future__ import annotations` を解釈できないため、
新しい構文には依存しないこと。
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

PROJ = Path("/work/gfsi/ufsi0002/bs2026-resolution")
SIF = "/home1/gfsi/ufsi0002/bird_detection/bird_python.sif"
DATA_ROOT = "/work/gfsi/ufsi0002/bs2026-raptor/data_root"
# ⭐ 査読指摘 A-3 の group split (元画像・動画系列をまたがない分割) で学習し直せるように
#    環境変数で差し替えられるようにした。**既定は従来どおり**なので既存の再実行に影響しない。
#      SPLITS_DIR=/work/gfsi/ufsi0002/bs2026-raptor/data/splits_group python3 train/run_phase2.py ...
SPLITS_DIR = os.environ.get("SPLITS_DIR",
                            "/work/gfsi/ufsi0002/bs2026-raptor/data/splits")
PRETRAIN_DIR = "/work/gfsi/ufsi0002/bs2026-raptor/weights"

MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]
RESOLUTIONS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]

# v2 の学習上限。初版 (80 / 30) では 24/84 が上限に張り付いたため引き上げた
EPOCHS = {"mnv4": 160, "effb0": 160, "resnet50": 160, "vit_small": 160,
          "dinov2_l": 60, "dinov3_l": 60}
PATIENCE = {"mnv4": 20, "effb0": 20, "resnet50": 20, "vit_small": 25,
            "dinov2_l": 10, "dinov3_l": 10}

# 224 における 1 epoch の実測秒 (Phase 0/1 の実測値)
BASE_SEC_PER_EPOCH = {"mnv4": 10.0, "effb0": 9.0, "resnet50": 10.0,
                      "vit_small": 9.5, "dinov2_l": 36.0, "dinov3_l": 36.0}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def est_cost(model, res):
    """所要時間の見積り (秒)。スケジューリング順を決めるためだけに使う。

    解像度の 2 乗にほぼ比例するとみなすが、CNN は CPU 前処理が律速で解像度を下げても
    あまり縮まないことが実測で分かったので、下限を高めに置く。
    """
    scale = (res / 224.0) ** 2
    floor = 0.35 if not model.startswith("dino") else 0.08
    return BASE_SEC_PER_EPOCH[model] * EPOCHS[model] * max(scale, floor)


def run_job(gpu, model, res, seed, models_dir, dry, num_workers):
    tag = "%s_r%d" % (model, res)
    out = PROJ / models_dir / tag
    logfile = PROJ / "logs" / ("%s_s%d.log" % (tag, seed))
    out.mkdir(parents=True, exist_ok=True)
    cmd = (
        "CUDA_VISIBLE_DEVICES=%d apptainer exec --nv "
        "--env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env CUDA_VISIBLE_DEVICES=%d "
        "--bind /work --bind /home1 %s "
        "python3 train/train_res.py --model %s --res %d --mode B "
        "--data_root %s --splits_dir %s --pretrain_dir %s "
        "--seed %d --epochs %d --patience %d --num_workers %d --output %s"
        % (gpu, gpu, SIF, model, res, DATA_ROOT, SPLITS_DIR, PRETRAIN_DIR,
           seed, EPOCHS[model], PATIENCE[model], num_workers, out)
    )
    if dry:
        log("[dry] gpu%d %s: %s..." % (gpu, tag, cmd[:80]))
        return {"tag": tag, "rc": 0, "sec": 0.0}
    t0 = time.time()
    with open(str(logfile), "w") as f:
        rc = subprocess.run(["bash", "-c", cmd], cwd=str(PROJ),
                            stdout=f, stderr=subprocess.STDOUT).returncode
    return {"tag": tag, "model": model, "res": res, "seed": seed, "gpu": gpu,
            "rc": rc, "sec": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs="*", type=int, default=[0, 1, 2, 3])
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--resolutions", nargs="*", type=int, default=RESOLUTIONS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models_dir", default="models_s42",
                   help="seed ごとに分ける (models_s42 / models_s43 / ...)")
    p.add_argument("--force", action="store_true", help="完了済みでも再実行する")
    p.add_argument("--dry", action="store_true", help="コマンドを表示するだけ")
    p.add_argument("--num_workers", type=int, default=10,
                   help="DataLoader のワーカ数。CNN は解像度を下げても CPU 前処理が律速なので、"
                        "GPU 数 x num_workers がノードの 48 コアを超えない範囲で増やすと効く")
    p.add_argument("--progress_name", default="phase2_progress.json")
    args = p.parse_args()

    jobs = []
    skipped = []
    for m in args.models:
        for r in args.resolutions:
            done_marker = PROJ / args.models_dir / ("%s_r%d" % (m, r)) / "test_metrics.json"
            if done_marker.exists() and not args.force:
                skipped.append("%s_r%d" % (m, r))
                continue
            jobs.append((m, r))

    jobs.sort(key=lambda x: -est_cost(x[0], x[1]))
    total_est = sum(est_cost(j[0], j[1]) for j in jobs)
    log("seed=%d models_dir=%s jobs=%d skipped=%d gpus=%s"
        % (args.seed, args.models_dir, len(jobs), len(skipped), args.gpus))
    log("見積り総計算時間 %.1f GPU-h -> %d 並列で約 %.1f h"
        % (total_est / 3600, len(args.gpus), total_est / 3600 / len(args.gpus)))
    log("学習上限: CNN/ViT-S %d epoch (patience %d) / ViT-L %d epoch (patience %d)"
        % (EPOCHS["effb0"], PATIENCE["effb0"], EPOCHS["dinov2_l"], PATIENCE["dinov2_l"]))

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results = []
    res_lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                model, res = q.get_nowait()
            except queue.Empty:
                log("gpu%d 終了 (キュー空)" % gpu)
                return
            log("gpu%d start %s_r%d (残り %d)" % (gpu, model, res, q.qsize()))
            r = run_job(gpu, model, res, args.seed, args.models_dir, args.dry, args.num_workers)
            with res_lock:
                results.append(r)
                json.dump(results, open(str(PROJ / "results" / args.progress_name), "w"),
                          indent=2, ensure_ascii=False)
            log("gpu%d done  %s %s %.1fs"
                % (gpu, r["tag"], "ok" if r["rc"] == 0 else "NG rc=%d" % r["rc"], r["sec"]))
            q.task_done()

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(g,), daemon=False) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ng = [r for r in results if r["rc"] != 0]
    log("=== Phase 2 完了 seed=%d: %d 本 / 失敗 %d 本 / 実時間 %.2f h ==="
        % (args.seed, len(results), len(ng), (time.time() - t0) / 3600))
    for r in ng:
        log("  失敗: %s rc=%d -> logs/%s_s%d.log を確認" % (r["tag"], r["rc"], r["tag"], args.seed))
    (PROJ / "logs" / ("phase2_s%d.done" % args.seed)).touch()


if __name__ == "__main__":
    main()
