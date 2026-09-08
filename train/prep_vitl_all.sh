#!/usr/bin/env bash
# ViT-L のエッジ実測を全 14 水準へ広げるための ONNX 下ごしらえ (HPC 側).
#
# 既に 32/64/128/224 は用意済みなので、残る 10 水準 (16/48/80/96/112/144/160/176/192/208)
# を同じ手順で作る。手順はモデルで少し違う:
#
#   共通  : 条件 B の重みから FP16 ONNX を書き出し、onnxruntime の基本最適化に通し、
#           transformer ブロック境界で 4 分割し、実画像の前処理済み入力を作る
#   v2 のみ: p3 をさらに 2 分割する (後半だけ FP32 にするため)
#   v3 のみ: 4 パートそれぞれに ORT 基本最適化を掛け直す (これをしないと定数エンジンになる)
#
# usage: bash prep_vitl_all.sh <res...>
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
APP="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
cd "$PROJ" || exit 1
rm -f logs/prep_vitl_all.done

for R in "$@"; do
    for M in dinov2_l dinov3_l; do
        tag="${M}_r${R}"
        base="${tag}_fp16_sim"
        echo "[$(date '+%H:%M:%S')] ===== ${tag} ====="

        if [ ! -f "$E/onnx/${tag}_fp16.onnx" ]; then
            $APP python3 train/export_onnx.py --models_dir "$PROJ/models_s42" \
                --models "$M" --resolutions "$R" --half --suffix _fp16 \
                --output_dir "$E/onnx" > "$PROJ/logs/prep_${tag}_export.log" 2>&1
            echo "  [1] FP16 ONNX $(stat -c %s $E/onnx/${tag}_fp16.onnx 2>/dev/null | awk '{printf "%.0f MB", $1/1e6}')"
        else
            echo "  [1] FP16 ONNX 既存"
        fi

        if [ ! -f "$E/onnx/${base}.onnx" ]; then
            $APP python3 train/fold_if.py "$E/onnx/${tag}_fp16.onnx" "$E/onnx/${base}.onnx" \
                > "$PROJ/logs/prep_${tag}_fold.log" 2>&1
            echo "  [2] ORT 基本最適化 rc=$?"
        else
            echo "  [2] 最適化済み"
        fi

        if [ ! -f "$E/onnx_split/${base}_p3.onnx" ]; then
            $APP python3 train/split_onnx.py "$E/onnx/${base}.onnx" "$E/onnx_split" --nparts 4 \
                > "$PROJ/logs/prep_${tag}_split.log" 2>&1
            echo "  [3] 4 分割 rc=$?"
        else
            echo "  [3] 分割済み"
        fi

        if [ "$M" = "dinov2_l" ]; then
            if [ ! -f "$E/onnx_split_p3/${base}_p3_p1.onnx" ]; then
                $APP python3 train/split_onnx.py "$E/onnx_split/${base}_p3.onnx" "$E/onnx_split_p3" --nparts 2 \
                    > "$PROJ/logs/prep_${tag}_splitp3.log" 2>&1
                echo "  [4] p3 を 2 分割 rc=$?"
            else
                echo "  [4] p3 分割済み"
            fi
        else
            if [ ! -f "$E/onnx_split/${base}_p3_ort.onnx" ]; then
                $APP python3 train/ort_parts.py "$E/onnx_split" "$base" 4 \
                    > "$PROJ/logs/prep_${tag}_ortparts.log" 2>&1
                echo "  [4] パート単位 ORT 最適化 rc=$?"
            else
                echo "  [4] パート最適化済み"
            fi
        fi

        if [ ! -f "$E/inputs/real_${tag}.fp16.bin" ]; then
            $APP python3 train/prep_inputs_vitl.py --models "$M" --res "$R" \
                --data_root "$DR" --splits_dir "$SP" --out_dir "$E/inputs" \
                > "$PROJ/logs/prep_${tag}_inputs.log" 2>&1
            echo "  [5] 入力 $(stat -c %s $E/inputs/real_${tag}.fp16.bin 2>/dev/null | awk '{printf "%.0f MB", $1/1e6}')"
        else
            echo "  [5] 入力既存"
        fi
    done
done
touch logs/prep_vitl_all.done
echo "[$(date '+%H:%M:%S')] ===== 準備完了 ====="
