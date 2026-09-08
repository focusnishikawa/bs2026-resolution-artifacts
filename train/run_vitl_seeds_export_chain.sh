#!/usr/bin/env bash
# ViT-L 配備精度 (査読指摘 A-4) のための ONNX 書き出しチェーン (HPC 側).
# seed 43-71 の 29 シードを順に書き出す。1 シードあたり実測 34 分・約 27 GB。
# 書き出しが済んだシードには $E/vitl_seeds/s<NN>/export.done が立ち、
# fgpu0 側のチェーンがそれを見てエンジン構築へ進む (fgpu0 が先行しないようにする)。
# usage: nohup setsid bash train/run_vitl_seeds_export_chain.sh > logs/vitl_export_chain.log 2>&1 &
set -u
PROJ=/work/gfsi/ufsi0002/bs2026-resolution
cd "$PROJ" || exit 1
T0=$(date +%s)
echo "===== ViT-L ONNX 書き出しチェーン 開始 $(date '+%F %T') ====="
for s in $(seq 43 71); do
    if [ -f "/home1/gfsi/ufsi0002/bs2026-resolution-edge/vitl_seeds/s${s}/export.done" ]; then
        echo "[skip] seed $s (書き出し済み)"; continue
    fi
    echo "########## seed $s 開始 $(date '+%F %T') ##########"
    bash train/prep_vitl_seed.sh "$s"
    echo "########## seed $s 終了 rc=$? $(date '+%F %T') ##########"
done
echo "===== 全 29 シード完了 $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/3600 )) 時間 ====="
touch logs/vitl_export_chain.done
