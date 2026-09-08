#!/usr/bin/env bash
# 評価の欠損を埋めるリトライ.
#
# 評価経路には散発的な SIGSEGV があり (ViT 系の N=16 で特に起きやすいが再現性はない)、
# プロセスが死ぬと同じ担当分の残りが実行されずに欠ける。--skip_existing で完了分は
# スキップされるので、同じコマンドを繰り返せば欠損は埋まっていく。
#
# 落ちても他が進むよう GPU は 1 プロセスずつ直列に回す (速度より確実性を優先)。
#
# 起動: nohup setsid bash train/retry_eval.sh > logs/retry_eval.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
APP_GPU="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
APP_CPU="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
RES_ALL="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
RES_NO16="32 48 64 80 96 112 128 144 160 176 192 208 224"

cd "$PROJ" || exit 1

count() {  # $1=mode $2=seed
    ls "${PROJ}/results/T1_cond$1_s$2" 2>/dev/null \
        | grep -cE "^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r[0-9]+\.json$"
}

for attempt in 1 2 3 4 5 6; do
    all_ok=1
    for seed in 42 43 44; do
        for mode in A B; do
            n=$(count "$mode" "$seed")
            if [ "$n" -ge 84 ]; then continue; fi
            all_ok=0
            echo "[$(date '+%H:%M:%S')] attempt ${attempt}: cond${mode} s${seed} = ${n}/84 -> 再実行"
            MD="${PROJ}/models_s${seed}"
            OUT="${PROJ}/results/T1_cond${mode}_s${seed}"
            if [ "$mode" = "A" ]; then RES_USE="$RES_ALL"; else RES_USE="$RES_NO16"; fi

            # GPU 分 (1 プロセスずつ、モデル単位で分けて 1 本落ちても他が進むようにする)
            for mo in mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l; do
                CUDA_VISIBLE_DEVICES=0 ${APP_GPU} python3 train/eval_sweep.py \
                    --mode "$mode" --models "$mo" --resolutions ${RES_USE} \
                    --models_dir "$MD" --output_dir "$OUT" \
                    --data_root "$DR" --splits_dir "$SP" \
                    --skip_existing --torch_threads 8 \
                    --summary_name "summary_retry_${mo}.json" \
                    >> "${PROJ}/logs/retry_${mode}_${seed}.log" 2>&1
            done

            # 条件 B の N=16 は CPU (CUDA だとトークン 1 個で落ちる)
            if [ "$mode" = "B" ]; then
                for mo in mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l; do
                    CUDA_VISIBLE_DEVICES= ${APP_CPU} python3 train/eval_sweep.py \
                        --mode B --models "$mo" --resolutions 16 \
                        --models_dir "$MD" --output_dir "$OUT" \
                        --data_root "$DR" --splits_dir "$SP" \
                        --skip_existing --torch_threads 8 --batch_size 32 \
                        --summary_name "summary_retry_cpu_${mo}.json" \
                        >> "${PROJ}/logs/retry_${mode}_${seed}.log" 2>&1
                done
            fi
            echo "[$(date '+%H:%M:%S')]   -> $(count "$mode" "$seed")/84"
        done
    done
    if [ "$all_ok" -eq 1 ]; then
        echo "[$(date '+%H:%M:%S')] 全 6 系列が 84/84 に到達 (attempt ${attempt} 時点で欠損なし)"
        break
    fi
done

echo "[$(date '+%H:%M:%S')] ===== リトライ終了 ====="
for seed in 42 43 44; do
    for mode in A B; do
        echo "  cond${mode} seed${seed}: $(count "$mode" "$seed") / 84"
    done
done
touch "${PROJ}/logs/retry_eval.done"
