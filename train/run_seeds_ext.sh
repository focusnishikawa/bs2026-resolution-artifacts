#!/usr/bin/env bash
# 拡張 seed について「条件 B 学習 -> 条件 A/B 評価 -> 欠損リトライ」を seed 単位で完結させる.
#
# 経緯: 図 3 の帯 (シード間ばらつき) が 3 seed では統計サンプリングとして薄いため、
#       seed を増やす。既存の 42/43/44 と完全に同じレシピ・学習上限・評価経路を使い、
#       違いは seed だけにする (統制を崩さない)。
#
# 設計:
#   - **seed 単位で学習と評価を閉じる**。途中で止めてもその seed までは完成品になる。
#     (run_all_seeds.sh + run_eval_v2.sh は全 seed の学習を終えてから評価する作りで、
#      長時間ジョブでは中断時に評価が丸ごと欠ける)
#   - 各段は既存成果物をスキップするので、同じコマンドで何度でも再開できる。
#   - 条件 A の基準モデルは条件 B の r224 を流用するため、追加学習は要らない。
#
# ⚠️ N=16 の振り分け (実測に基づく。run_eval_v2.sh と同じ):
#     条件 B の N=16 は入力 16x16 で CUDA が SIGSEGV -> CPU で実行
#     条件 A の N=16 は入力 224 なので GPU で実行 (CPU だと ViT が 224 入力で落ちる)
#
# 起動 (hpciaiss1):
#   cd /work/gfsi/ufsi0002/bs2026-resolution
#   nohup setsid bash train/run_seeds_ext.sh 45 46 47 48 49 50 51 52 53 54 \
#       > logs/seeds_ext_master.log 2>&1 &
#
# 監視: tail -f logs/seeds_ext_master.log
# 完了: logs/seeds_ext.done
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
APP_GPU="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
APP_CPU="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"

RES_ALL="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
RES_NO16="32 48 64 80 96 112 128 144 160 176 192 208 224"
MODELS_ALL="mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l"

cd "$PROJ" || exit 1
mkdir -p logs results

if [ "$#" -eq 0 ]; then
    echo "usage: bash train/run_seeds_ext.sh <seed> [<seed> ...]"
    exit 1
fi
SEEDS="$*"

rm -f logs/seeds_ext.done

count() {  # $1=mode $2=seed  -> 完了した評価構成の数
    ls "${PROJ}/results/T1_cond$1_s$2" 2>/dev/null \
        | grep -cE "^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r[0-9]+\.json$"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 拡張 seed 開始: ${SEEDS} ====="

for seed in $SEEDS; do
    MD="${PROJ}/models_s${seed}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ########## seed ${seed} 開始 ##########"

    # ---- 1. 条件 B の学習 (84 構成。GPU 4 基へ動的割当) ----
    ntr=$(ls "${MD}"/*/test_metrics.json 2>/dev/null | wc -l)
    if [ "$ntr" -ge 84 ]; then
        echo "[$(date '+%H:%M:%S')] seed ${seed} 学習: 既に 84/84 -> skip"
    else
        echo "[$(date '+%H:%M:%S')] seed ${seed} 学習開始 (現在 ${ntr}/84)"
        python3 train/run_phase2.py \
            --seed "${seed}" \
            --models_dir "models_s${seed}" \
            --progress_name "p2_s${seed}.json" \
            --gpus 0 1 2 3 --num_workers 10 \
            > "logs/phase2_s${seed}.log" 2>&1
        echo "[$(date '+%H:%M:%S')] seed ${seed} 学習終了 (rc=$?) -> $(ls "${MD}"/*/test_metrics.json 2>/dev/null | wc -l)/84"
    fi

    # ---- 2. 条件 A / B の評価 (4 GPU 並列) ----
    for mode in A B; do
        OUT="${PROJ}/results/T1_cond${mode}_s${seed}"
        mkdir -p "$OUT"
        if [ "$(count "$mode" "$seed")" -ge 84 ]; then
            echo "[$(date '+%H:%M:%S')] seed ${seed} 条件 ${mode}: 既に 84/84 -> skip"
            continue
        fi
        if [ "$mode" = "A" ]; then RES_USE="$RES_ALL"; else RES_USE="$RES_NO16"; fi
        echo "[$(date '+%H:%M:%S')] seed ${seed} 条件 ${mode} 評価開始"
        rm -f logs/ev_${mode}_${seed}_gpu*.done

        i=0
        for models in "mnv4 dinov2_l" "effb0 dinov3_l" "resnet50" "vit_small"; do
            tag="ev_${mode}_${seed}_gpu${i}"
            nohup setsid bash -c "
                CUDA_VISIBLE_DEVICES=${i} ${APP_GPU} python3 train/eval_sweep.py \
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

        # 条件 B の N=16 は CPU
        if [ "$mode" = "B" ]; then
            CUDA_VISIBLE_DEVICES= ${APP_CPU} python3 train/eval_sweep.py \
                --mode B --resolutions 16 \
                --models_dir "${MD}" --output_dir "${OUT}" \
                --data_root "${DR}" --splits_dir "${SP}" \
                --skip_existing --torch_threads 8 --batch_size 32 \
                --summary_name summary_cpu_r16.json \
                > "${PROJ}/logs/ev_B_${seed}_cpu16.log" 2>&1
        fi
        echo "[$(date '+%H:%M:%S')] seed ${seed} 条件 ${mode} 評価: $(count "$mode" "$seed")/84"
    done

    # ---- 3. 欠損リトライ (eval_sweep は dinov3_l で散発的に SIGSEGV する) ----
    for attempt in 1 2 3 4 5; do
        need=0
        for mode in A B; do
            [ "$(count "$mode" "$seed")" -lt 84 ] && need=1
        done
        [ "$need" -eq 0 ] && break
        echo "[$(date '+%H:%M:%S')] seed ${seed} リトライ ${attempt}"
        for mode in A B; do
            [ "$(count "$mode" "$seed")" -ge 84 ] && continue
            OUT="${PROJ}/results/T1_cond${mode}_s${seed}"
            if [ "$mode" = "A" ]; then RES_USE="$RES_ALL"; else RES_USE="$RES_NO16"; fi
            # 1 モデルずつ直列。1 本落ちても他は進む
            for mo in $MODELS_ALL; do
                CUDA_VISIBLE_DEVICES=0 ${APP_GPU} python3 train/eval_sweep.py \
                    --mode "$mode" --models "$mo" --resolutions ${RES_USE} \
                    --models_dir "$MD" --output_dir "$OUT" \
                    --data_root "$DR" --splits_dir "$SP" \
                    --skip_existing --torch_threads 8 \
                    --summary_name "summary_retry_${mo}.json" \
                    >> "${PROJ}/logs/retry_${mode}_${seed}.log" 2>&1
            done
            if [ "$mode" = "B" ]; then
                for mo in $MODELS_ALL; do
                    CUDA_VISIBLE_DEVICES= ${APP_CPU} python3 train/eval_sweep.py \
                        --mode B --models "$mo" --resolutions 16 \
                        --models_dir "$MD" --output_dir "$OUT" \
                        --data_root "$DR" --splits_dir "$SP" \
                        --skip_existing --torch_threads 8 --batch_size 32 \
                        --summary_name "summary_retry_cpu_${mo}.json" \
                        >> "${PROJ}/logs/retry_${mode}_${seed}.log" 2>&1
                done
            fi
        done
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ########## seed ${seed} 完了: condA $(count A "$seed")/84  condB $(count B "$seed")/84 ##########"
    touch "logs/seed_${seed}.done"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 全 seed 完了 ====="
for seed in $SEEDS; do
    echo "  seed ${seed}: 学習 $(ls "${PROJ}/models_s${seed}"/*/test_metrics.json 2>/dev/null | wc -l)/84  condA $(count A "$seed")/84  condB $(count B "$seed")/84"
done
touch logs/seeds_ext.done
