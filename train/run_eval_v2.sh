#!/usr/bin/env bash
# Phase 2 v2 の評価: 3 seed x 条件 A/B x 84 構成 = 504 評価を回す.
#
# 条件 A の基準モデルも v2 の 224 学習済みを使う (初版の models/ ではなく models_s<seed>/)。
# これで条件 A と条件 B が同一の学習上限・同一 seed で揃う。
#
# ⚠️ N=16 の扱い (実測に基づく分岐):
#   条件 B の N=16 は入力が 16x16 (patch16 ViT でトークン 1 個) で、**CUDA 上で SIGSEGV** する。
#     -> CPU で実行する。入力が小さいので CPU でも数十秒。
#   条件 A の N=16 は入力が 224 (トークン 196) なので CUDA では問題ない。
#     逆に **CPU で ViT を 224 入力で推論すると SIGSEGV** する (vit_small/dinov2_l/dinov3_l で確認)。
#     -> GPU で実行する。
#   つまり「小さい入力は CPU、大きい入力は GPU」で振り分ける。
#
# 起動: nohup setsid bash train/run_eval_v2.sh > logs/eval_v2_master.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
APP_BASE="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
APP_CPU="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"

RES_ALL="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
RES_NO16="32 48 64 80 96 112 128 144 160 176 192 208 224"

cd "$PROJ" || exit 1
mkdir -p logs results

for seed in 42 43 44; do
    MD="${PROJ}/models_s${seed}"
    for mode in A B; do
        OUT="${PROJ}/results/T1_cond${mode}_s${seed}"
        mkdir -p "$OUT"
        if [ "$mode" = "A" ]; then
            RES_USE="$RES_ALL"      # 条件 A は入力が常に 224 なので全水準を GPU で
        else
            RES_USE="$RES_NO16"     # 条件 B の N=16 だけ後段の CPU に回す
        fi
        echo "[$(date '+%H:%M:%S')] ===== seed ${seed} 条件 ${mode} 開始 ====="
        rm -f logs/ev_${mode}_${seed}_gpu*.done

        i=0
        for models in "mnv4 dinov2_l" "effb0 dinov3_l" "resnet50" "vit_small"; do
            tag="ev_${mode}_${seed}_gpu${i}"
            nohup setsid bash -c "
                CUDA_VISIBLE_DEVICES=${i} ${APP_BASE} python3 train/eval_sweep.py \
                    --mode ${mode} --models ${models} --resolutions ${RES_USE} \
                    --models_dir ${MD} --output_dir ${OUT} \
                    --data_root ${DR} --splits_dir ${SP} \
                    --skip_existing --torch_threads 8 --summary_name summary_gpu${i}.json \
                    > ${PROJ}/logs/${tag}.log 2>&1
                echo DONE_RC=\$? >> ${PROJ}/logs/${tag}.log
                touch ${PROJ}/logs/${tag}.done
            " > /dev/null 2>&1 < /dev/null &
            i=$((i + 1))
            sleep 2
        done

        while [ "$(ls logs/ev_${mode}_${seed}_gpu*.done 2>/dev/null | wc -l)" -lt 4 ]; do
            sleep 20
        done
        echo "[$(date '+%H:%M:%S')] seed ${seed} 条件 ${mode}: GPU 分 完了"

        if [ "$mode" = "B" ]; then
            CUDA_VISIBLE_DEVICES= ${APP_CPU} python3 train/eval_sweep.py \
                --mode B --resolutions 16 \
                --models_dir "${MD}" --output_dir "${OUT}" \
                --data_root "${DR}" --splits_dir "${SP}" \
                --skip_existing --torch_threads 8 --batch_size 32 \
                --summary_name summary_cpu_r16.json \
                > "${PROJ}/logs/ev_B_${seed}_cpu16.log" 2>&1
            echo "[$(date '+%H:%M:%S')] seed ${seed} 条件 B: CPU(N=16) 完了 rc=$?"
        fi

        n=$(ls "${OUT}" | grep -cE "^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r[0-9]+\.json$")
        echo "[$(date '+%H:%M:%S')] ===== seed ${seed} 条件 ${mode} 完了: ${n} / 84 構成 ====="
    done
done

echo "[$(date '+%H:%M:%S')] ===== 全評価 完了 ====="
for seed in 42 43 44; do
    for mode in A B; do
        n=$(ls "${PROJ}/results/T1_cond${mode}_s${seed}" 2>/dev/null | grep -cE "^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r[0-9]+\.json$")
        echo "  cond${mode} seed${seed}: ${n} / 84"
    done
done
touch logs/eval_v2.done
