#!/usr/bin/env bash
# Phase 4b 修正: 条件 B の Stage1/Stage2 を各解像度の特徴量で学習し直す.
#
# ⚠️ 元の実装の不備:
#   条件 B では backbone を N で再学習し特徴も N で抽出したのに、
#   Stage1/Stage2 の LightGBM は 224 学習済み (models_v6_flat40k) を流用していた。
#   特徴分布が変わっているのに前段だけ旧モデルという不整合で、
#   condB の Stage1 が N=16/32 で 0.4994 (ランダム) に張り付いていた。
#
# ここで前段も条件 B の特徴量で学習し直し、前段・後段を揃える。
# Stage3+4 統合版は既に条件 B の特徴で学習済みなので再利用する。
#
# LightGBM のみなので GPU は不要。SIF が OMP_NUM_THREADS=1 を固定するため
# 明示的に上書きすること (これを忘れると 1 コアで数時間かかる)。
#
# 起動: nohup setsid bash run_v6_condB_stage12.sh > logs/v6_condB_s12_master.log 2>&1 &
set -u

B=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
P=/home1/gfsi/ufsi0002/fgpu0_v6_bench/pylibs_x86
A=/home1/gfsi/ufsi0002/apptainer/bin/apptainer
S=/home1/gfsi/ufsi0002/sif/bird_phase5.sif
C2T=$B/configs/class_to_taxon_v5.json
LOG=$B/logs/v6_condB_s12.log
SCALES="16 32 48 64 80 96 112 128 144 160 176 192 208"
THREADS=16          # 3 本並列でも 48 コアに収まる本数

cd "$B" || exit 1
LGB=("$A" exec --bind /home1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH="$P" --env OMP_NUM_THREADS=$THREADS "$S")

echo "=== START $(date '+%F %T') 条件B Stage1/2 再学習 ===" | tee -a "$LOG"

train_one() {
    local S_=$1
    local TAG="cb${S_}"
    local FEAT=$B/data/features_condB_full_y_${TAG}_cleaned.npz
    local MDIR=$B/models_condB/lgb_${TAG}
    [ -f "$FEAT" ] || { echo "  [miss] $FEAT" | tee -a "$LOG"; return; }
    if [ -f "$MDIR/stage2.txt" ]; then echo "  [skip] stage1/2 ${TAG}" | tee -a "$LOG"; return; fi
    echo "--- stage1,2 ${TAG} ---" | tee -a "$LOG"
    "${LGB[@]}" python3 scripts/train_hierarchy_lgb.py \
        --features "$FEAT" --output-dir "$MDIR" \
        --stages 1,2 --num-rounds 40000 --early-stopping 100 \
        --num-threads $THREADS --class-to-taxon "$C2T" >> "$LOG" 2>&1
    echo "  stage1,2 ${TAG} rc=$?" | tee -a "$LOG"
}

# 3 本ずつ並列 (48 コア / 16 スレッド)
i=0
for S_ in $SCALES; do
    train_one "$S_" &
    i=$((i + 1))
    [ $((i % 3)) -eq 0 ] && wait
done
wait

echo "=== 前段が揃ったので推論をやり直す ===" | tee -a "$LOG"
for S_ in $SCALES; do
    TAG="cb${S_}"
    MDIR=$B/models_condB/lgb_${TAG}
    [ -f "$MDIR/stage2.txt" ] || { echo "  [skip infer] ${TAG} (stage2 無し)" | tee -a "$LOG"; continue; }
    [ -f "$MDIR/stage34.txt" ] || { echo "  [skip infer] ${TAG} (stage34 無し)" | tee -a "$LOG"; continue; }
    for SET in bird nonbird; do
        OUT=$B/preds/v6_condBfix_s34r500_gt_${SET}_${TAG}.csv
        [ -f "$OUT" ] && continue
        # m12 も m34 も条件 B のモデルを使う (ここが修正点)
        "${LGB[@]}" python3 scripts/infer_s34_rounds.py \
            "$MDIR" "$MDIR" \
            "$B/data/features_condB_gt_${SET}_${TAG}.npz" "$C2T" "$OUT" 500 >> "$LOG" 2>&1
        echo "  infer ${TAG} ${SET} rc=$? -> $(basename "$OUT")" | tee -a "$LOG"
    done
done

echo "=== DONE $(date '+%F %T') ===" | tee -a "$LOG"
touch "$B/logs/v6_condB_s12.done"
