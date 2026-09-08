#!/usr/bin/env bash
# ViT-L の配備構成 ONNX を **シードごと** に書き出す (**m001 版**).
#
# train/prep_vitl_seed.sh の m001 移植。**書き出しの中身は 1 行も変えていない**
# (変えると seed 42 と別のグラフを測ることになる)。違うのはパスだけである:
#   - コード      /home1/gfsi/ufsi0002/bs2026-resolution-code   (hpciaiss1 の /work から複製)
#   - ckpt        /home1/gfsi/ufsi0002/bs2026-resolution-ckpt/models_s<NN>
#   - 出力 ONNX   /home1/gfsi/ufsi0002/bs2026-resolution-edge/vitl_seeds/s<NN>  (fgpu0 と共有)
#
# ⛔ hpciaiss1 は他セッションが計算に使っているので触らない。m001 は GPU を持たないが、
#    export_onnx.py は `map_location="cpu"` の CPU 処理なので支障はない。
# ⚠️ m001 は外部ネットワーク不通なので HF_HUB_OFFLINE=1 が必須 (キャッシュは /home1 にある)。
#
# ⭐ 由来を seed 42 と揃えることが最重要である:
#   DINOv2-L: FP16 ONNX -> **ORT を通さず** 4 分割 -> p3 をさらに 2 分割
#   DINOv3-L: FP16 ONNX -> fold_if -> 4 分割 -> **パート単位 ORT 最適化**
#
# usage:
#   bash train/prep_vitl_seed_m001.sh 43            # 全 14 水準
#   bash train/prep_vitl_seed_m001.sh 43 112        # 1 水準だけ (パイロット用)
set -u

SEED="${1:?usage: prep_vitl_seed_m001.sh <seed> [res...]}"; shift
RES="${*:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"

PROJ="${PROJ:-/home1/gfsi/ufsi0002/bs2026-resolution-code}"
STAGE="${STAGE:-/home1/gfsi/ufsi0002/bs2026-resolution-ckpt}"
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
S="$E/vitl_seeds/s${SEED}"
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APPBIN="${APPBIN:-/home1/gfsi/ufsi0002/apptainer/bin/apptainer}"
APP="$APPBIN exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /home1 $SIF"
MDIR="$STAGE/models_s${SEED}"
LOG="$PROJ/logs/vitl_s${SEED}"

cd "$PROJ" || exit 1
[ -x "$APPBIN" ] || { echo "[abort] apptainer が無い: $APPBIN"; exit 1; }
[ -d "$MDIR" ] || { echo "[abort] チェックポイントが無い: $MDIR"; exit 1; }
mkdir -p "$S/onnx" "$S/onnx_split_raw" "$S/onnx_split_raw_p3" "$S/onnx_split" "$LOG"
rm -f "$S/export.done"

ok=0; ng=0
for R in $RES; do
    # ---------------- DINOv2-L: 生の 4 分割 + p3 の 2 分割 ----------------
    tag="dinov2_l_r${R}"
    if [ ! -s "$S/onnx_split_raw_p3/${tag}_fp16_p3_p1.onnx" ]; then
        echo "[$(date '+%H:%M:%S')] s${SEED} ${tag}"
        if [ ! -s "$S/onnx/${tag}_fp16.onnx" ]; then
            $APP python3 train/export_onnx.py --models_dir "$MDIR" \
                --models dinov2_l --resolutions "$R" --half --suffix _fp16 \
                --output_dir "$S/onnx" > "$LOG/${tag}_export.log" 2>&1 \
                || { echo "  [NG] export"; ng=$((ng+1)); continue; }
        fi
        $APP python3 train/split_onnx.py "$S/onnx/${tag}_fp16.onnx" "$S/onnx_split_raw" --nparts 4 \
            > "$LOG/${tag}_split.log" 2>&1 || { echo "  [NG] split4"; ng=$((ng+1)); continue; }
        $APP python3 train/split_onnx.py "$S/onnx_split_raw/${tag}_fp16_p3.onnx" "$S/onnx_split_raw_p3" --nparts 2 \
            > "$LOG/${tag}_splitp3.log" 2>&1 || { echo "  [NG] splitp3"; ng=$((ng+1)); continue; }
        echo "  [OK] dinov2_l 5 パート"
        ok=$((ok+1))
    else
        echo "[skip] s${SEED} ${tag}"
    fi

    # ---------------- DINOv3-L: fold_if -> 4 分割 -> パート単位 ORT ----------------
    tag="dinov3_l_r${R}"
    base="${tag}_fp16_sim"
    if [ ! -s "$S/onnx_split/${base}_p3_ort.onnx" ]; then
        echo "[$(date '+%H:%M:%S')] s${SEED} ${tag}"
        if [ ! -s "$S/onnx/${tag}_fp16.onnx" ]; then
            $APP python3 train/export_onnx.py --models_dir "$MDIR" \
                --models dinov3_l --resolutions "$R" --half --suffix _fp16 \
                --output_dir "$S/onnx" > "$LOG/${tag}_export.log" 2>&1 \
                || { echo "  [NG] export"; ng=$((ng+1)); continue; }
        fi
        if [ ! -s "$S/onnx/${base}.onnx" ]; then
            $APP python3 train/fold_if.py "$S/onnx/${tag}_fp16.onnx" "$S/onnx/${base}.onnx" \
                > "$LOG/${tag}_fold.log" 2>&1 || { echo "  [NG] fold_if"; ng=$((ng+1)); continue; }
        fi
        $APP python3 train/split_onnx.py "$S/onnx/${base}.onnx" "$S/onnx_split" --nparts 4 \
            > "$LOG/${tag}_split.log" 2>&1 || { echo "  [NG] split4"; ng=$((ng+1)); continue; }
        $APP python3 train/ort_parts.py "$S/onnx_split" "$base" 4 \
            > "$LOG/${tag}_ortparts.log" 2>&1 || { echo "  [NG] ort_parts"; ng=$((ng+1)); continue; }
        echo "  [OK] dinov3_l 4 パート"
        ok=$((ok+1))
    else
        echo "[skip] s${SEED} ${tag}"
    fi

    # 分割前の中間 ONNX は容量が大きい (1 水準あたり 2.4 GB) ので落とす。
    if [ -s "$S/onnx_split_raw_p3/dinov2_l_r${R}_fp16_p3_p1.onnx" ] \
       && [ -s "$S/onnx_split/dinov3_l_r${R}_fp16_sim_p3_ort.onnx" ]; then
        rm -f "$S/onnx/dinov2_l_r${R}_fp16.onnx" \
              "$S/onnx/dinov3_l_r${R}_fp16.onnx" "$S/onnx/dinov3_l_r${R}_fp16_sim.onnx" \
              "$S/onnx_split/dinov3_l_r${R}_fp16_sim_p"[0-3]".onnx"
    fi
done

echo "[$(date '+%H:%M:%S')] s${SEED} 完了: 作成 ${ok} / 失敗 ${ng}  ($(du -sh "$S" 2>/dev/null | cut -f1))"

# ⚠️⚠️ マーカーは「失敗 0」ではなく「**全 14 水準がそろっている**」ことで立てる
FULL="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
have=0; miss=""
for R in $FULL; do
    if [ -s "$S/onnx_split_raw_p3/dinov2_l_r${R}_fp16_p3_p1.onnx" ] \
       && [ -s "$S/onnx_split/dinov3_l_r${R}_fp16_sim_p3_ort.onnx" ]; then
        have=$((have+1))
    else
        miss="$miss $R"
    fi
done
echo "  そろった水準: ${have} / 14"
if [ "$ng" -eq 0 ] && [ "$have" -eq 14 ]; then
    touch "$S/export.done"
    echo "  マーカー: $S/export.done"
else
    rm -f "$S/export.done"
    echo "  マーカーは立てない (失敗 ${ng} / 未了水準:${miss:- なし})"
fi
exit "$ng"
