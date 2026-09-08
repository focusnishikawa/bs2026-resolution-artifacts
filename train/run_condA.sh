#!/usr/bin/env bash
# 条件 A (劣化耐性) スイープを 4 GPU へ分割して実行する.
# モデル間は完全に独立なので、モデル集合を GPU へ割り当てるだけで並列化できる。
# 重い ViT-L 2 種を別 GPU に散らし、CNN と組ませて所要時間を均す。
#
# 起動: bash train/run_condA.sh
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
OUT=${PROJ}/results/T1_condA

APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
cd "$PROJ" || exit 1
mkdir -p "$OUT" logs

launch() {
    local gpu="$1"; shift
    local models="$*"
    local tag=$(echo "$models" | tr ' ' '-')
    echo "[$(date '+%H:%M:%S')] gpu=${gpu} models=${models}"
    nohup setsid bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} ${APP} python3 train/eval_sweep.py \
            --mode A --models ${models} \
            --models_dir ${PROJ}/models --output_dir ${OUT} \
            --data_root ${DR} --splits_dir ${SP} \
            --skip_existing --summary_name summary_gpu${gpu}.json \
            > ${PROJ}/logs/condA_gpu${gpu}.log 2>&1
        echo DONE_RC=\$? >> ${PROJ}/logs/condA_gpu${gpu}.log
        touch ${PROJ}/logs/condA_gpu${gpu}.done
    " > /dev/null 2>&1 < /dev/null &
}

rm -f "${PROJ}"/logs/condA_gpu*.done

launch 0 mnv4 dinov2_l
sleep 3
launch 1 effb0 dinov3_l
sleep 3
launch 2 resnet50
sleep 3
launch 3 vit_small

echo "=== 4 プロセス投入完了。完了待ち ==="
n=0
while [ "$(ls "${PROJ}"/logs/condA_gpu*.done 2>/dev/null | wc -l)" -lt 4 ]; do
    sleep 30
    n=$((n + 1))
    [ $((n % 10)) -eq 0 ] && echo "[$(date '+%H:%M:%S')] waiting ... ($((n / 2)) min)"
done

echo "=== 条件 A スイープ 完了 $(date) ==="
ls -1 "${OUT}"/*.json | wc -l
touch "${PROJ}/logs/condA.done"
