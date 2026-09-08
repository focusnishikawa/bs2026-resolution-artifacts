#!/usr/bin/env bash
# ViT-L を N=32 以外の解像度でもエッジ実測できるように、ONNX の下ごしらえをする (HPC 側).
#
# 報告書 4.4 の限界「ViT-L の Orin 実測は N=32 のみ」を埋めるため。
# 3.3 の結論「解像度を削っても速くならない」が ViT-L の分割チェーンでも成り立つかを見る。
#
# 手順 (N=32 で確立した経路をそのまま流用):
#   1. 条件 B の学習済み重みから **FP16 の ONNX** を書き出す (FP32 1.2GB は Orin でパース不能)
#   2. onnxruntime の基本最適化に通す (DINOv3-L の If を畳む。DINOv2-L でも同じ経路を使う)
#   3. transformer ブロック境界で 4 分割する
#   4. DINOv2-L は p3 が FP16 で壊れるので、p3 をさらに 2 分割しておく
#   5. 入力依存性の確認に使う実画像の前処理済みバイナリを作る
#
# usage: bash prep_vitl_res.sh <model> <res...>
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
APP="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
MODEL="${1:-dinov2_l}"
shift || true
RESLIST="${*:-64 128 224}"
cd "$PROJ" || exit 1

for R in $RESLIST; do
    echo "===== ${MODEL} @ N=${R} ====="
    tag="${MODEL}_r${R}"

    # 1. FP16 ONNX
    if [ ! -f "$E/onnx/${tag}_fp16.onnx" ]; then
        echo "  [1] FP16 ONNX を書き出す"
        $APP python3 train/export_onnx.py --models_dir "$PROJ/models_s42" \
            --models "$MODEL" --resolutions "$R" --half --suffix _fp16 \
            --output_dir "$E/onnx" > "$PROJ/logs/prep_${tag}_export.log" 2>&1
        echo "      $(ls -la $E/onnx/${tag}_fp16.onnx 2>/dev/null | awk '{printf "%.0f MB", $5/1e6}')"
    else
        echo "  [1] FP16 ONNX は既存"
    fi

    # 2. ORT 基本最適化 (If を畳む / 標準 opset に留める)
    if [ ! -f "$E/onnx/${tag}_fp16_sim.onnx" ]; then
        echo "  [2] ORT 基本最適化"
        $APP python3 train/fold_if.py "$E/onnx/${tag}_fp16.onnx" "$E/onnx/${tag}_fp16_sim.onnx" \
            > "$PROJ/logs/prep_${tag}_fold.log" 2>&1
        tail -2 "$PROJ/logs/prep_${tag}_fold.log" | sed 's/^/      /'
    else
        echo "  [2] 最適化済み ONNX は既存"
    fi

    # 3. 4 分割
    if [ ! -f "$E/onnx_split/${tag}_p3.onnx" ]; then
        echo "  [3] 4 分割"
        $APP python3 train/split_onnx.py "$E/onnx/${tag}_fp16_sim.onnx" "$E/onnx_split" --nparts 4 \
            > "$PROJ/logs/prep_${tag}_split.log" 2>&1
        grep -aE "^\[part|^\[done" "$PROJ/logs/prep_${tag}_split.log" | sed 's/^/      /'
    else
        echo "  [3] 分割済み"
    fi

    # 4. DINOv2-L は p3 をさらに 2 分割 (FP16 で壊れる区間を切り出すため)
    if [ "$MODEL" = "dinov2_l" ] && [ ! -f "$E/onnx_split_p3/${tag}_p3_p1.onnx" ]; then
        echo "  [4] p3 を 2 分割"
        $APP python3 train/split_onnx.py "$E/onnx_split/${tag}_p3.onnx" "$E/onnx_split_p3" --nparts 2 \
            > "$PROJ/logs/prep_${tag}_splitp3.log" 2>&1
        grep -aE "^\[part|^\[done" "$PROJ/logs/prep_${tag}_splitp3.log" | sed 's/^/      /'
    fi

    # 5. 実画像の前処理済み入力 (入力依存性の確認に使う)
    if [ ! -f "$E/inputs/real_${tag}.fp16.bin" ]; then
        echo "  [5] 入力バイナリ"
        $APP python3 train/prep_inputs_vitl.py --models "$MODEL" --res "$R" \
            --data_root "$DR" --splits_dir "$SP" --out_dir "$E/inputs" \
            > "$PROJ/logs/prep_${tag}_inputs.log" 2>&1
        ls -la "$E/inputs/real_${tag}.fp16.bin" 2>/dev/null | awk '{printf "      %.1f MB\n", $5/1e6}'
    else
        echo "  [5] 入力バイナリは既存"
    fi
done
echo "===== 準備完了 ====="
