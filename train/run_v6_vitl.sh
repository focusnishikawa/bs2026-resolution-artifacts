#!/usr/bin/env bash
# Phase 4b-ViT: v6 カスケードの backbone を自己教師あり ViT-L に差し替えた条件 B.
#
# 代表 4 水準 (32 / 64 / 128 / 224) x DINOv2-L / DINOv3-L = 8 本。
# ResNet50 版 (run_v6_condB.sh) と同じ工程だが、特徴次元が 2048 -> 1024 になるため
# LightGBM は全段 (stage1/2 と Stage3+4 統合版) を条件 B の特徴で学習し直す。
#
# ⚠️ DINOv2 は patch14 なので入力は 14 の倍数へ丸まる (32->28, 64->70, 128->126, 224->224)。
#    丸め後の寸法は meta に記録され、特徴抽出側も V6_INPUT_RES で同じ値を使う。
#
# 起動: GPU_ID=0 nohup setsid bash run_v6_vitl.sh <model_id> <解像度…> > logs/x.log 2>&1 &
set -u

B=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
P=/home1/gfsi/ufsi0002/fgpu0_v6_bench/pylibs_x86
A=/home1/gfsi/ufsi0002/apptainer/bin/apptainer
SIF_GPU=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
SIF_LGB=/home1/gfsi/ufsi0002/sif/bird_phase5.sif
TORCH_CACHE=/home1/gfsi/ufsi0002/bird_detection/torch_cache
C2T=$B/configs/class_to_taxon_v5.json
HIER=$B/data/hierarchy_labels_v5b_full_y.csv
GTB=$B/data/hierarchy_labels_gt_84_y.csv
GTN=$B/data/hierarchy_labels_gt_nonbird_y.csv
MODEL="${1:-dinov3_l}"
SCALES="${2:-32 64 128 224}"
GPU_ID="${GPU_ID:-0}"
LGB_THREADS="${LGB_THREADS:-16}"
LOG=$B/logs/v6_vitl_${MODEL}_${GPU_ID}.log

cd "$B" || exit 1
mkdir -p logs preds data models_vitl

# ⚠️ apptainer はホスト側の環境変数を自動では渡さない。V6_INPUT_RES のように
# コンテナ内の python が読む値は必ず --env で明示すること
# (コマンド前置の `V6_INPUT_RES=32 apptainer exec …` では中に届かず、
#  ViT-L が img_size=224 で構築されて入力と形状不一致になった)
GPU_BASE=("$A" exec --nv --bind /home1 --bind /work --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 \
     --env TORCH_HOME="$TORCH_CACHE" --env CUDA_VISIBLE_DEVICES="$GPU_ID")
GPU=("${GPU_BASE[@]}" "$SIF_GPU")
LGB=("$A" exec --bind /home1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH="$P" \
     --env OMP_NUM_THREADS="$LGB_THREADS" "$SIF_LGB")

echo "=== START $(date '+%F %T') ${MODEL} scales=[${SCALES}] gpu=${GPU_ID} ===" | tee -a "$LOG"

for S in $SCALES; do
    TAG="${MODEL}_r${S}"
    CKDIR=$B/models_vitl/ft_${TAG}
    echo "########## ${MODEL} N=${S} ##########" | tee -a "$LOG"

    # (1) backbone: head + 最終 2 ブロックのみ学習 (v6 の head+layer4 に相当)
    if [ -f "$CKDIR/best.pt" ]; then
        echo "  [skip] backbone ${TAG}" | tee -a "$LOG"
    else
        echo "--- (1) finetune ${TAG} ---" | tee -a "$LOG"
        "${GPU[@]}" python3 scripts/finetune_vitl_v6.py \
            --hierarchy-csv "$HIER" --output-dir "$CKDIR" \
            --model-id "$MODEL" --downsample-to "$S" --native \
            --epochs 20 --batch-size 64 --num-workers 12 \
            --lr-head 5e-4 --lr-layer4 5e-5 >> "$LOG" 2>&1
        echo "  finetune rc=$?" | tee -a "$LOG"
    fi
    [ -f "$CKDIR/best.pt" ] || { echo "  [NG] backbone 無し" | tee -a "$LOG"; continue; }

    # (2) 特徴抽出
    # ⚠️ DINOv2 は patch14 なので入力寸法が丸まる (32->28, 64->70, 128->126)。
    # モデル (V6_INPUT_RES) と transform (--downsample-to) の両方に
    # **丸め後の値**を渡さないと「Input height doesn't match model」で落ちる。
    if [ "$MODEL" = "dinov2_l" ]; then
        SR=$(python3 -c "p=14; r=$S; print(max(p, p*round(r/p)))")
    else
        SR=$S
    fi
    echo "  [res] N=$S -> 入力寸法 $SR (${MODEL})" | tee -a "$LOG"
    for SET in full_y gt_bird gt_nonbird; do
        case $SET in
            full_y)     CSV=$HIER; EXTRA="" ;;
            gt_bird)    CSV=$GTB;  EXTRA="" ;;
            gt_nonbird) CSV=$GTN;  EXTRA="--target-nonbird 840" ;;
        esac
        OUT=$B/data/features_vitl_${SET}_${TAG}.npz
        [ -f "$OUT" ] && { echo "  [skip] feat ${SET}"; continue; }
        echo "--- (2) extract ${SET} ${TAG} ---" | tee -a "$LOG"
        "${GPU_BASE[@]}" --env V6_INPUT_RES="$SR" "$SIF_GPU" \
            python3 scripts/extract_features_res_v6.py \
            --hierarchy-csv "$CSV" --output "$OUT" --model "$MODEL" \
            --checkpoint "$CKDIR/best.pt" --downsample-to "$SR" --native \
            --batch-size 128 --num-workers 12 --device cuda $EXTRA >> "$LOG" 2>&1
        echo "  extract ${SET} rc=$?" | tee -a "$LOG"
    done

    # (2.5) Stage4 ラベル整理 (75 -> 71 クラス)
    FULL=$B/data/features_vitl_full_y_${TAG}.npz
    CLEAN=$B/data/features_vitl_full_y_${TAG}_cleaned.npz
    if [ ! -f "$CLEAN" ]; then
        "${LGB[@]}" python3 scripts/cleanup_stage4_labels.py --input "$FULL" --output "$CLEAN" >> "$LOG" 2>&1
        echo "  cleanup rc=$?" | tee -a "$LOG"
    fi

    # (3) LightGBM 全段 (特徴 1024 次元なので stage1/2 も必須)
    MDIR=$B/models_vitl/lgb_${TAG}
    mkdir -p "$MDIR"
    if [ ! -f "$MDIR/stage2.txt" ]; then
        echo "--- (3a) stage1,2 ${TAG} ---" | tee -a "$LOG"
        "${LGB[@]}" python3 scripts/train_hierarchy_lgb.py \
            --features "$CLEAN" --output-dir "$MDIR" --stages 1,2 \
            --num-rounds 40000 --early-stopping 100 --num-threads "$LGB_THREADS" \
            --class-to-taxon "$C2T" >> "$LOG" 2>&1
        echo "  stage1,2 rc=$?" | tee -a "$LOG"
    fi
    if [ ! -f "$MDIR/stage34.txt" ]; then
        echo "--- (3b) stage34 統合版 ${TAG} ---" | tee -a "$LOG"
        "${LGB[@]}" python3 scripts/train_stage34_merged.py \
            --features "$CLEAN" --ref-models "$B/models_v6_flat40k" --out-dir "$MDIR" >> "$LOG" 2>&1
        echo "  stage34 rc=$?" | tee -a "$LOG"
    fi

    # (4) 推論 (500 木打ち切り)
    for SET in bird nonbird; do
        OUT=$B/preds/v6_vitl_s34r500_gt_${SET}_${TAG}.csv
        [ -f "$OUT" ] && continue
        [ -f "$MDIR/stage34.txt" ] || break
        "${LGB[@]}" python3 scripts/infer_s34_rounds.py \
            "$MDIR" "$MDIR" "$B/data/features_vitl_gt_${SET}_${TAG}.npz" \
            "$C2T" "$OUT" 500 >> "$LOG" 2>&1
        echo "  infer ${SET} rc=$? -> $(basename "$OUT")" | tee -a "$LOG"
    done
done

echo "=== DONE $(date '+%F %T') ${MODEL} ===" | tee -a "$LOG"
touch "$B/logs/v6_vitl_${MODEL}_${GPU_ID}.done"
