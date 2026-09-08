#!/usr/bin/env bash
# Phase 4b: v6 カスケードの **条件 B (解像度別ネイティブ再学習)**.
#
# Phase 4a は既存の出荷モデルに縮小画像を入れるだけ (条件 A) だった。ここでは
# 各解像度で backbone から学習し直し、T1 (猛禽 6 種) で見つけた
# 「**CNN は低解像度で再学習すると +34〜46 pt 改善する**」が
# 5 段カスケード全体でも成立するかを確かめる。
#
# 1 解像度あたりの流れ:
#   (1) ResNet50 を N でネイティブ学習 (full_y 151,539 枚)
#   (2) その重みで full_y と GT の特徴を N ネイティブで抽出
#   (3) LightGBM を再学習: stage1/2 と **Stage3+4 統合版 (74 クラス)**
#       ※ v6 の結論が「両段は統合してよい」なので統合版を主軸に置く
#   (4) GT を推論 (500 木打ち切り)
#
# ⚠️ SIF は OMP_NUM_THREADS=1 を固定しており LightGBM がこれを自分の num_threads より
#    優先する。48 コア機で 1 コアしか使わず初回は 2 時間 25 分かけて終わらなかった前例が
#    あるため、必ず --env OMP_NUM_THREADS=32 を渡す。
#
# 起動: nohup setsid bash run_v6_condB.sh <解像度リスト> > logs/v6_condB_master.log 2>&1 &
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
LOG=$B/logs/v6_condB_${GPU_ID:-0}.log
SCALES="${1:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"

cd "$B" || exit 1
mkdir -p logs preds data models_condB

# 並列実行時は GPU_ID / LGB_THREADS を外から与える。
# LightGBM を 4 本同時に走らせると 4x32=128 スレッドで 48 コアを超えるため、
# 並列時は LGB_THREADS=12 (4x12=48) に絞ること。
GPU_ID="${GPU_ID:-0}"
LGB_THREADS="${LGB_THREADS:-32}"
GPU=("$A" exec --nv --bind /home1 --env PYTHONNOUSERSITE=1 --env TORCH_HOME="$TORCH_CACHE" --env CUDA_VISIBLE_DEVICES="$GPU_ID" "$SIF_GPU")
LGB=("$A" exec --bind /home1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH="$P" --env OMP_NUM_THREADS="$LGB_THREADS" "$SIF_LGB")

echo "=== START $(date '+%F %T') 条件B scales=[$SCALES] ===" | tee -a "$LOG"

for S in $SCALES; do
    TAG="cb${S}"
    CKDIR=$B/models_condB/resnet50_ft_${TAG}
    echo "########## N=${S} ##########" | tee -a "$LOG"

    # (1) backbone をネイティブ N で学習
    if [ -f "$CKDIR/best.pt" ]; then
        echo "  [skip] backbone ${TAG}" | tee -a "$LOG"
    else
        echo "--- (1) finetune N=${S} ---" | tee -a "$LOG"
        "${GPU[@]}" python3 scripts/finetune_res_v6.py \
            --hierarchy-csv "$HIER" --output-dir "$CKDIR" \
            --downsample-to "$S" --native \
            --epochs 20 --batch-size 128 --num-workers 16 >> "$LOG" 2>&1
        echo "  finetune rc=$?" | tee -a "$LOG"
    fi
    [ -f "$CKDIR/best.pt" ] || { echo "  [NG] backbone 無し -> N=${S} を飛ばす" | tee -a "$LOG"; continue; }

    # (2) 特徴抽出 (full_y / GT bird / GT nonbird) をネイティブ N で
    for SET in full_y gt_bird gt_nonbird; do
        case $SET in
            full_y)     CSV=$HIER ;;
            gt_bird)    CSV=$GTB ;;
            gt_nonbird) CSV=$GTN ;;
        esac
        OUT=$B/data/features_condB_${SET}_${TAG}.npz
        [ -f "$OUT" ] && { echo "  [skip] feat ${SET} ${TAG}"; continue; }
        echo "--- (2) extract ${SET} N=${S} ---" | tee -a "$LOG"
        # 非鳥集合は --target-nonbird を渡さないと 1 枚も抽出されない (kept=0 になる)
        EXTRA=""
        [ "$SET" = "gt_nonbird" ] && EXTRA="--target-nonbird 840"
        V6_INPUT_RES=$S "${GPU[@]}" python3 scripts/extract_features_res_v6.py \
            --hierarchy-csv "$CSV" --output "$OUT" --model resnet50 \
            --checkpoint "$CKDIR/best.pt" --downsample-to "$S" --native \
            --batch-size 256 --num-workers 16 --device cuda $EXTRA >> "$LOG" 2>&1
        echo "  extract ${SET} rc=$?" | tee -a "$LOG"
    done

    # (2.5) Stage4 ラベルの整理 (75 -> 71 クラス)
    #   重複名 3 種の統合と Looney_Birds (176 枚の誤ラベル) の除去。
    #   これを飛ばすと統合版の 74 クラス定義とラベル空間がずれ、
    #   LightGBM が "Label must be in [0, 74), but found 74" で落ちる。
    FULL=$B/data/features_condB_full_y_${TAG}.npz
    CLEAN=$B/data/features_condB_full_y_${TAG}_cleaned.npz
    if [ ! -f "$CLEAN" ]; then
        echo "--- (2.5) cleanup_stage4_labels ${TAG} (75 -> 71) ---" | tee -a "$LOG"
        "${LGB[@]}" python3 scripts/cleanup_stage4_labels.py \
            --input "$FULL" --output "$CLEAN" >> "$LOG" 2>&1
        echo "  cleanup rc=$?" | tee -a "$LOG"
    fi

    # (3) LightGBM を再学習 (stage1/2 + Stage3+4 統合版)
    MDIR=$B/models_condB/lgb_${TAG}
    if [ -f "$MDIR/stage34.txt" ]; then
        echo "  [skip] lgb ${TAG}" | tee -a "$LOG"
    else
        echo "--- (3) LightGBM ${TAG} ---" | tee -a "$LOG"
        mkdir -p "$MDIR"
        "${LGB[@]}" python3 scripts/train_stage34_merged.py \
            --features "$CLEAN" \
            --ref-models "$B/models_v6_flat40k" --out-dir "$MDIR" >> "$LOG" 2>&1
        echo "  lgb rc=$?" | tee -a "$LOG"
    fi

    # (4) GT を推論 (500 木)
    for SET in bird nonbird; do
        OUT=$B/preds/v6_condB_s34r500_gt_${SET}_${TAG}.csv
        [ -f "$OUT" ] && continue
        [ -f "$MDIR/stage34.txt" ] || { echo "  [NG] lgb 無し"; break; }
        "${LGB[@]}" python3 scripts/infer_s34_rounds.py \
            "$B/models_v6_flat40k" "$MDIR" \
            "$B/data/features_condB_gt_${SET}_${TAG}.npz" "$C2T" "$OUT" 500 >> "$LOG" 2>&1
        echo "  infer ${SET} rc=$? -> $(basename "$OUT")" | tee -a "$LOG"
    done
done

echo "=== DONE $(date '+%F %T') ===" | tee -a "$LOG"
touch "$B/logs/v6_condB_${GPU_ID:-0}.done"
