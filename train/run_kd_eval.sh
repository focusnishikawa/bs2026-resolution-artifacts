#!/usr/bin/env bash
# 蒸留生徒 (effb0 @224) の劣化耐性を 14 解像度で評価する (条件 A).
#
# 比較相手は同じ effb0 の CE-only 3 seed (既存 results/T1_condA)。
# 学習条件は損失以外すべて揃えてあるので、この差がそのまま「蒸留の低解像度耐性への寄与」になる。
#
# ⚠️ GPU は 2,3 のみ。GPU0-1 は他セッションが占有中。
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
cd "$PROJ" || exit 1
rm -f logs/kd_eval.done

i=0
for seed in 42 43 44; do
    gpu=$(( 2 + i % 2 ))
    out="${PROJ}/results/T1_condA_kd_s${seed}"
    log="${PROJ}/logs/kd_eval_s${seed}.log"
    mkdir -p "$out"
    echo "[$(date '+%H:%M:%S')] eval seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} ${APP} python3 train/eval_sweep.py \
        --mode A --models effb0 \
        --models_dir "${PROJ}/models_kd_s${seed}" --output_dir "$out" \
        --data_root ${DR} --splits_dir ${SP} \
        --summary_name "summary_kd_s${seed}.json" > "$log" 2>&1
    echo "  rc=$? $(grep -ac '^\[eval\]' "$log" 2>/dev/null) 構成"
    i=$((i + 1))
done
touch logs/kd_eval.done
echo "[$(date '+%H:%M:%S')] 完了"
