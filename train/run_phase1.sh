#!/usr/bin/env bash
# Phase 1: 基準モデル 6 種を N=224 で学習する (条件 A の基準 = 条件 B の N=224 水準を兼ねる).
#
# 第 1 陣 (GPU 0-3): mnv4 / effb0 / resnet50 / vit_small  ... 80 epoch, batch 256
# 第 2 陣 (GPU 0-1): dinov2_l / dinov3_l                  ... 30 epoch, batch 64, fp32
#
# 全ジョブが同一 seed (42) / 同一 aug / 同一 split で走ることが条件 A の前提。
# 起動は必ず bash で行う: bash train/run_phase1.sh
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
PW=/work/gfsi/ufsi0002/bs2026-raptor/weights
RES=224

APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
cd "$PROJ" || exit 1
mkdir -p logs models results

launch() {
    local gpu="$1" model="$2"
    local tag="${model}_r${RES}"
    local out="${PROJ}/models/${tag}"
    local log="${PROJ}/logs/${tag}.log"
    mkdir -p "$out"
    echo "[$(date '+%H:%M:%S')] launch gpu=${gpu} ${tag}"
    nohup setsid bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} ${APP} python3 train/train_res.py \
            --model ${model} --res ${RES} --mode B \
            --data_root ${DR} --splits_dir ${SP} --pretrain_dir ${PW} \
            --output ${out} > ${log} 2>&1
        echo DONE_RC=\$? >> ${log}
        touch ${PROJ}/logs/${tag}.done
    " > /dev/null 2>&1 < /dev/null &
}

wait_for() {
    local n=0
    for tag in "$@"; do
        while [ ! -f "${PROJ}/logs/${tag}.done" ]; do
            sleep 30
            n=$((n + 1))
            if [ $((n % 20)) -eq 0 ]; then
                echo "[$(date '+%H:%M:%S')] waiting ${tag} ... ($((n / 2)) min)"
            fi
        done
        echo "[$(date '+%H:%M:%S')] done ${tag}"
    done
}

rm -f "${PROJ}"/logs/*_r${RES}.done

echo "=== 第 1 陣 (CNN 3 + ViT-S) ==="
launch 0 mnv4
sleep 3
launch 1 effb0
sleep 3
launch 2 resnet50
sleep 3
launch 3 vit_small
wait_for mnv4_r${RES} effb0_r${RES} resnet50_r${RES} vit_small_r${RES}

echo "=== 第 2 陣 (ViT-L 2) ==="
launch 0 dinov2_l
sleep 3
launch 1 dinov3_l
wait_for dinov2_l_r${RES} dinov3_l_r${RES}

echo "=== Phase 1 学習 完了 $(date) ==="
for m in mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l; do
    f="${PROJ}/models/${m}_r${RES}/test_metrics.json"
    if [ -f "$f" ]; then
        python3 -c "
import json; d=json.load(open('$f'))
print(f\"  {d['model']:10s} test_acc={d.get('test_acc',0):.4f} macro_rec={d.get('macro_recall',0):.4f} \"
      f\"best_val={d.get('best_val_acc',0):.4f} ep={d.get('epochs_run',0)} sec={d.get('train_sec',0)}\")
"
    else
        echo "  ${m}: test_metrics.json なし (失敗の可能性)"
    fi
done
touch "${PROJ}/logs/phase1.done"
