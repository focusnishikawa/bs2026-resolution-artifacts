#!/usr/bin/env bash
# 査読指摘 A-3: group split (元画像・動画系列をまたがない分割) での再学習・再評価.
#
# 全 84 構成 x 30 シードの再学習は前回実績 5.46 時間/シード = 約 7 日かかる。レビューは
# 「広い探索を探索的結果、group split で再実験した候補群を確認的結果と明示する」ことを
# 許容しているので、**選択が入れ替わりうる境界の 14 構成**に絞る (ユーザー判断 2026-09-05)。
#
#   ResNet50  N=32/64/96/112/128/160/176/208/224 … 選択表の 0.80-0.93 とその近傍
#   DINOv2-L  N=64/144/160                        … 0.95/0.96 と 10 ms 予算の最良
#   ViT-S/16  N=112/224                           … サーバ精度で選ぶと選ばれる構成
#
# ⚠️ 学習・評価とも既存成果物を上書きしない別ディレクトリへ書く:
#     models_group_s<seed>/ , results/G_cond{A,B}_s<seed>/
# ⚠️ 分割は SPLITS_DIR 環境変数で差し替える (run_phase2.py / eval_sweep.py とも対応済み)。
# ⚠️ 条件 A は N=16 も要るが、境界に N=16 は無いので条件 B と同じ水準だけを回す。
#
# 起動:
#   nohup setsid bash train/run_group_split_boundary.sh 42 43 ... > logs/group_boundary_master.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SPG=/work/gfsi/ufsi0002/bs2026-raptor/data/splits_group
APP_GPU="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"

R50="32 48 64 80 96 112 128 160 176 208 224"
V2L="64 144 160"
VITS="112 224"

cd "$PROJ" || exit 1
mkdir -p logs results
[ -d "$SPG" ] || { echo "[abort] group split が無い: $SPG (先に train/make_group_split.py)"; exit 1; }
[ "$#" -ge 1 ] || { echo "usage: bash train/run_group_split_boundary.sh <seed>..."; exit 1; }
SEEDS="$*"
rm -f logs/group_boundary.done

echo "[$(date '+%F %T')] ===== group split 境界構成 開始: seeds ${SEEDS} ====="
echo "  分割: $SPG"
echo "  構成: ResNet50 x $(echo $R50|wc -w) + DINOv2-L x $(echo $V2L|wc -w) + ViT-S/16 x $(echo $VITS|wc -w)"

for seed in $SEEDS; do
    MD="${PROJ}/models_group_s${seed}"
    echo "[$(date '+%F %T')] ########## seed ${seed} ##########"

    # ---- 1. 条件 B の学習 (境界構成のみ) ----
    for spec in "resnet50:$R50" "dinov2_l:$V2L" "vit_small:$VITS"; do
        m="${spec%%:*}"; rs="${spec#*:}"
        SPLITS_DIR="$SPG" python3 train/run_phase2.py \
            --seed "$seed" --models "$m" --resolutions $rs \
            --models_dir "models_group_s${seed}" \
            --progress_name "gb_${m}_s${seed}.json" \
            --gpus 0 1 2 3 --num_workers 10 \
            >> "logs/group_boundary_s${seed}.log" 2>&1
        echo "[$(date '+%H:%M:%S')]   学習 ${m} rc=$?  完了 $(ls ${MD}/*/test_metrics.json 2>/dev/null | wc -l)"
    done

    # ---- 2. 条件 A / B の評価 (4 GPU 並列) ----
    for mode in A B; do
        OUT="${PROJ}/results/G_cond${mode}_s${seed}"
        mkdir -p "$OUT"
        n=$(ls "$OUT" 2>/dev/null | grep -cE '^(resnet50|dinov2_l|vit_small)_r[0-9]+\.json$')
        if [ "$n" -ge 16 ]; then echo "[$(date '+%H:%M:%S')]   条件 ${mode}: 既に ${n}/16 -> skip"; continue; fi
        i=0
        for spec in "resnet50:$R50" "dinov2_l:$V2L" "vit_small:$VITS"; do
            m="${spec%%:*}"; rs="${spec#*:}"
            nohup setsid bash -c "
                CUDA_VISIBLE_DEVICES=${i} ${APP_GPU} python3 train/eval_sweep.py \
                    --mode ${mode} --models ${m} --resolutions ${rs} \
                    --models_dir ${MD} --output_dir ${OUT} \
                    --data_root ${DR} --splits_dir ${SPG} \
                    > logs/gb_ev_${mode}_${seed}_${m}.log 2>&1
                touch logs/gb_ev_${mode}_${seed}_${m}.done" > /dev/null 2>&1 &
            i=$((i+1))
        done
        # 3 本そろうまで待つ
        w=0
        while [ "$(ls logs/gb_ev_${mode}_${seed}_*.done 2>/dev/null | wc -l)" -lt 3 ]; do
            sleep 30; w=$((w+1))
            [ "$w" -gt 240 ] && { echo "[abort] 条件 ${mode} の評価が 2 時間で終わらない"; exit 1; }
        done
        rm -f logs/gb_ev_${mode}_${seed}_*.done
        echo "[$(date '+%H:%M:%S')]   条件 ${mode} 評価完了 $(ls "$OUT" | grep -cE '\.json$')/16"
    done
done
echo "[$(date '+%F %T')] ===== 全 seed 完了 ====="
touch logs/group_boundary.done
