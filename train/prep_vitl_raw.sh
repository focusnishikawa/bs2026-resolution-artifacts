#!/usr/bin/env bash
# DINOv2-L の分割 ONNX を **ORT 最適化を通さずに** 作る (N=32 と同じ経路).
#
# 経緯: 全 14 水準を測ったところ、DINOv2-L は N=32 だけ 16.4 ms で、両隣の N=16 (40.3 ms)・
# N=48 (41.0 ms) から大きく外れた。調べると N=32 のエンジンだけ **ORT 基本最適化を通す前の
# 生の ONNX** から分割していた。ORT を通すとグラフ構造が変わって TensorRT の MYELIN 融合が
# 効かなくなり、同じ計算でも 2.5 倍遅くなるらしい。
#
# DINOv2-L には畳むべき If が無いので ORT 最適化は不要である (必要だったのは DINOv3-L だけ)。
# よって全水準を N=32 と同じ「生の分割」で揃え直す。
#
# usage: bash prep_vitl_raw.sh <res...>
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
cd "$PROJ" || exit 1
rm -f logs/prep_vitl_raw.done

for R in "$@"; do
    tag="dinov2_l_r${R}"
    echo "[$(date '+%H:%M:%S')] ===== ${tag} (生の分割) ====="

    # FP16 ONNX は fp16_sim を作るときに書き出したものをそのまま使う
    if [ ! -f "$E/onnx/${tag}_fp16.onnx" ]; then
        echo "  [NG] $E/onnx/${tag}_fp16.onnx が無い"; continue
    fi

    if [ ! -f "$E/onnx_split_raw/${tag}_fp16_p3.onnx" ]; then
        $APP python3 train/split_onnx.py "$E/onnx/${tag}_fp16.onnx" "$E/onnx_split_raw" --nparts 4 \
            > "$PROJ/logs/prep_raw_${tag}_split.log" 2>&1
        echo "  [1] 4 分割 rc=$?"
    else
        echo "  [1] 分割済み"
    fi

    if [ ! -f "$E/onnx_split_raw_p3/${tag}_fp16_p3_p1.onnx" ]; then
        $APP python3 train/split_onnx.py "$E/onnx_split_raw/${tag}_fp16_p3.onnx" "$E/onnx_split_raw_p3" --nparts 2 \
            > "$PROJ/logs/prep_raw_${tag}_splitp3.log" 2>&1
        echo "  [2] p3 を 2 分割 rc=$?"
    else
        echo "  [2] p3 分割済み"
    fi
done
touch logs/prep_vitl_raw.done
echo "[$(date '+%H:%M:%S')] ===== 準備完了 ====="
