#!/usr/bin/env bash
# ViT-L 配備精度 (査読指摘 A-4) のエンジン構築+全数推論チェーン (fgpu0 側).
# HPC 側チェーンが立てる export.done を待ってから 1 シードずつ処理する。
# 1 シード実測 98 分・29 シードで約 47 時間。エンジンと ONNX は水準ごとに掃除する。
# usage: nohup setsid bash run_vitl_acc_chain.sh > logs/vitl_acc_chain.log 2>&1 &
set -u
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
T0=$(date +%s)
echo "===== ViT-L 配備精度チェーン 開始 $(date '+%F %T') ====="
for s in $(seq 43 71); do
    [ -f "logs/vitl_acc_s${s}.done" ] && { echo "[skip] seed $s (測定済み)"; continue; }
    # 書き出しを待つ (最大 6 時間)。HPC 側が 34 分/シードなので通常はすぐ立つ
    w=0
    while [ ! -f "vitl_seeds/s${s}/export.done" ]; do
        [ "$w" -ge 360 ] && { echo "[abort] seed $s の書き出しを 6 時間待ったが来ない"; exit 1; }
        [ $((w % 30)) -eq 0 ] && echo "[wait] seed $s の ONNX 待ち ($w 分)"
        sleep 60; w=$((w+1))
    done
    echo "########## seed $s 開始 $(date '+%F %T') ##########"
    bash run_vitl_acc_seed.sh "$s"
    echo "########## seed $s 終了 rc=$? $(date '+%F %T') ##########"
done
echo "===== 全 29 シード完了 $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/3600 )) 時間 ====="
touch logs/vitl_acc_chain.done
