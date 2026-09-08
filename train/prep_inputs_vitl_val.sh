#!/usr/bin/env bash
# ViT-L の **validation** 入力 bin を全 14 水準ぶん書き出す (**m001 版**).
#
# 2026-09-08 の再レビュー指摘 A-5 への対応。配備エンジンの精度は test だけを測っており、
# **構成選択に使う validation は ViT-L 級のみサーバ FP32 のまま**だった。
# 選択規則を配備エンジンで統一するために val 側の入力 bin を作る。
#
# ⚠️ 入力 bin はシードに依存しない (前処理だけで決まる) ので **1 回作れば 29 シードで使える**。
# ⚠️ dinov2_l は patch14 なので入力寸法が丸まる (例 N=144 -> 140)。
#    prep_inputs_vitl.py が resolve_input_res で丸めるので、ここでは触らない。
# ⚠️ 出力先は **/home1 (m001 と fgpu0 の共有 FS)** なので転送は不要である。
# ⛔ hpciaiss1 では動かさない (他セッションが計算に使用中)。
#
# usage:
#   nohup setsid bash train/prep_inputs_vitl_val.sh > logs/prep_inputs_vitl_val.log 2>&1 &
set -u

PROJ="${PROJ:-/home1/gfsi/ufsi0002/bs2026-resolution-code}"
DATA="${DATA:-/home1/gfsi/ufsi0002/bs2026-raptor-data}"
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
OUT="$E/inputs_val_vitl"
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APPBIN="${APPBIN:-/home1/gfsi/ufsi0002/apptainer/bin/apptainer}"
APP="$APPBIN exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /home1 $SIF"
RES="${*:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"

cd "$PROJ" || exit 1
[ -d "$DATA/data_root" ] || { echo "[abort] data_root が無い: $DATA/data_root"; exit 1; }
[ -s "$DATA/splits/val.csv" ] || { echo "[abort] splits/val.csv が無い: $DATA/splits"; exit 1; }
mkdir -p "$OUT" logs
T0=$(date +%s)
echo "===== ViT-L val 入力 bin 書き出し 開始 $(date '+%F %T') ====="
echo "  ホスト: $(hostname) / 出力先: $OUT / 水準: $RES"

ok=0; ng=0
for R in $RES; do
    if [ -s "$OUT/real_dinov2_l_r${R}.fp16.bin" ] && [ -s "$OUT/real_dinov3_l_r${R}.fp16.bin" ]; then
        echo "  [skip] N=${R} (両モデルとも作成済み)"
        ok=$((ok+1)); continue
    fi
    t0=$(date +%s)
    if $APP python3 train/prep_inputs_vitl.py \
            --data_root "$DATA/data_root" --splits_dir "$DATA/splits" \
            --split val --models dinov2_l dinov3_l --res "$R" --out_dir "$OUT" \
            > "logs/prep_inputs_vitl_val_r${R}.log" 2>&1; then
        printf "  [ok  ] N=%-3d %4ds\n" "$R" "$(( $(date +%s)-t0 ))"
        ok=$((ok+1))
    else
        echo "  [FAIL] N=${R}"; tail -3 "logs/prep_inputs_vitl_val_r${R}.log"
        ng=$((ng+1))
    fi
done

# ⚠️ マーカーは「失敗 0」ではなく **全 28 本がそろったこと**で立てる
have=0; miss=""
for R in 16 32 48 64 80 96 112 128 144 160 176 192 208 224; do
    for M in dinov2_l dinov3_l; do
        if [ -s "$OUT/real_${M}_r${R}.fp16.bin" ]; then have=$((have+1)); else miss="$miss ${M}_r${R}"; fi
    done
done
echo "  そろった入力: ${have} / 28"
echo "===== 完了 $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/60 )) 分 (ok ${ok} / ng ${ng}) ====="
if [ "$have" -eq 28 ]; then
    touch "$OUT/inputs_val_vitl.done"
    echo "  マーカー: $OUT/inputs_val_vitl.done"
    exit 0
fi
rm -f "$OUT/inputs_val_vitl.done"
echo "  ⚠ マーカーを立てない (未了:${miss})"
exit 1
