#!/usr/bin/env bash
# Phase 4a-2: v6 の **Stage3+4 統合版 (74 クラス単一分類器)** を 14 解像度で推論する.
#
# v6 論文の結論は「Stage 3 (猛禽目) と Stage 4 (種) は統合してよい」だった:
#   統合により保護対象種アラートの precision 0.765 -> 0.929 (誤報 4 -> 1)、
#   IUCN 正解率 0.9452 -> 0.9512、モデル 2 本 -> 1 本、実機 latency も同等。
#   木は 500 本打ち切りが最良のバランス (LightGBM 段 p95 33.05 ms)。
# したがって解像度スイープも統合版を主軸に置く。分離版 (run_v6_res14.sh) と対で読む。
#
# ★検証したい仮説:
#   統合の利点は「他 3 目 (コンドル目/ハヤブサ目/フクロウ目) が受け皿として働き、
#   タカ目でない個体を種へ誤って割り当てにくくなる」ことにあった (目で棄却 68 -> 137)。
#   低解像度では目の判別自体が崩れるため、**この受け皿機構が壊れて統合の優位が消える**
#   可能性がある。解像度スイープはそれを直接確かめられる。
#
# 起動: nohup setsid bash run_v6_s34_res14.sh > logs/v6_s34_res14_master.log 2>&1 &
set -u

B=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
P=/home1/gfsi/ufsi0002/fgpu0_v6_bench/pylibs_x86
A=/home1/gfsi/ufsi0002/apptainer/bin/apptainer
S=/home1/gfsi/ufsi0002/sif/bird_phase5.sif
C2T=$B/configs/class_to_taxon_v5.json
M12=$B/models_v6_flat40k        # stage1 / stage2 はこちらを使う
M34=$B/models_v6_stage34        # 統合版 74 クラス
NI=500                          # 木の打ち切り本数 (v6 で最良と結論)
LOG=$B/logs/v6_s34_res14.log

cd "$B" || exit 1
mkdir -p preds logs
RUN=("$A" exec --bind /home1 --env PYTHONNOUSERSITE=1 --env PYTHONPATH="$P" --env OMP_NUM_THREADS=32 "$S")

echo "=== START $(date '+%F %T') 統合版 (500 木) x 14 解像度 ===" | tee -a "$LOG"

for SC in 16 32 48 64 80 96 112 128 144 160 176 192 208 224; do
    for SET in bird nonbird; do
        if [ "$SC" = "224" ]; then
            FEAT="$B/data/features_v6_gt_${SET}_r50_ft.npz"     # native は ds 接尾辞なし
            TAG="native"
        else
            FEAT="$B/data/features_v6_gt_${SET}_r50_ft_ds${SC}.npz"
            TAG="ds${SC}"
        fi
        OUT="$B/preds/v6_s34r${NI}_gt_${SET}_${TAG}.csv"
        if [ ! -f "$FEAT" ]; then echo "  [miss] $FEAT" | tee -a "$LOG"; continue; fi
        if [ -f "$OUT" ]; then echo "  [skip] $(basename "$OUT")" | tee -a "$LOG"; continue; fi
        echo "--- s34 ${TAG} ${SET} ---" | tee -a "$LOG"
        "${RUN[@]}" python3 scripts/infer_s34_rounds.py \
            "$M12" "$M34" "$FEAT" "$C2T" "$OUT" "$NI" >> "$LOG" 2>&1
        echo "  rc=$? -> $(basename "$OUT") $(wc -l < "$OUT" 2>/dev/null || echo 0) 行" | tee -a "$LOG"
    done
done

echo "=== DONE $(date '+%F %T') 出力 $(ls $B/preds/v6_s34r${NI}_gt_*.csv 2>/dev/null | wc -l) 本 ===" | tee -a "$LOG"
touch "$B/logs/v6_s34_res14.done"
