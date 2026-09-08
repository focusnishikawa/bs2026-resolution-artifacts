#!/usr/bin/env bash
# Phase 2 v2: 学習上限を引き上げた条件 B を seed 42/43/44 で順に回す (84 構成 x 3 = 252 本).
#
# 各 seed が GPU 4 基を使い切るので seed 間は逐次実行する。
# seed ごとに models_s<seed>/ へ出力するため、評価側は --models_dir を差し替えるだけで済む。
# 中断しても test_metrics.json の有無でスキップされるので、同じコマンドで再開できる。
#
# 起動: nohup setsid bash train/run_all_seeds.sh > logs/all_seeds_master.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
cd "$PROJ" || exit 1
mkdir -p logs results

for seed in 42 43 44; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== seed ${seed} 開始 ====="
    python3 train/run_phase2.py \
        --seed "${seed}" \
        --models_dir "models_s${seed}" \
        --progress_name "p2_s${seed}.json" \
        --gpus 0 1 2 3 --num_workers 10 \
        > "logs/phase2_s${seed}.log" 2>&1
    rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== seed ${seed} 終了 (rc=${rc}) ====="
    tail -3 "logs/phase2_s${seed}.log"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 全 seed 完了 ====="
for seed in 42 43 44; do
    n=$(ls "models_s${seed}"/*/test_metrics.json 2>/dev/null | wc -l)
    echo "  seed ${seed}: ${n} / 84 構成"
done
touch logs/phase2_all_seeds.done
