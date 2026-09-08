#!/usr/bin/env bash
# 30 シードの Orin 実配備精度を測るため、seed 43-71 の CNN 4 種 x 14 解像度を ONNX へ書き出す.
#
# 背景: 査読指摘 A1「配備精度をサーバ FP32 精度で代用している」への対応。
#   seed 42 の 1 シードでは、ViT-S/16 が N>=96 で FP16 化により最大 -10.4 ポイント
#   劣化することが分かった。表 5 を 30 シードの**配備精度**で作り直すために、
#   残り 29 シードのエンジンを作って全数推論する。
#
# ⚠️ ViT-L は対象外。分割チェーンなので 1 シードあたり 126 パートあり、
#    30 シードでは 3,780 エンジンになって現実的でない。ViT-L は一致率 97.9-100% で
#    半精度の影響が小さいことを seed 42 で実証済みなので、そちらは据え置く。
#
# 出力: /home1/.../bs2026-resolution-edge/onnx_seeds/s<NN>/*.onnx  (1 シードあたり約 3.1 GB)
# usage: bash train/export_onnx_30seed.sh [開始seed] [終了seed]
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
OUT=/home1/gfsi/ufsi0002/bs2026-resolution-edge/onnx_seeds
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP=/home1/gfsi/ufsi0002/apptainer/bin/apptainer
S0="${1:-43}"; S1="${2:-71}"

cd "$PROJ" || exit 1
mkdir -p logs "$OUT"
echo "===== ONNX 書き出し seed ${S0}-${S1} $(date '+%F %T') ====="

n_ok=0; n_skip=0; n_ng=0
for s in $(seq "$S0" "$S1"); do
    d="$OUT/s$s"
    # 56 本そろっていれば作り直さない (中断しても同じコマンドで再開できる)
    if [ -d "$d" ] && [ "$(ls "$d"/*.onnx 2>/dev/null | wc -l)" -eq 56 ]; then
        echo "[skip] s$s (56 本そろい)"; n_skip=$((n_skip+1)); continue
    fi
    if [ ! -d "$PROJ/models_s$s" ]; then
        echo "[miss] s$s (models_s$s が無い)"; n_ng=$((n_ng+1)); continue
    fi
    mkdir -p "$d"
    t0=$(date +%s)
    "$APP" exec --env PYTHONNOUSERSITE=1 -B /work,/home1 "$SIF" \
        python3 train/export_onnx.py --models_dir "models_s$s" --output_dir "$d" \
        --models mnv4 effb0 resnet50 vit_small --skip_existing \
        > "logs/export_onnx_s$s.log" 2>&1
    rc=$?
    n=$(ls "$d"/*.onnx 2>/dev/null | wc -l)
    printf "[%s] s%-3s rc=%d %3ds  %2d/56 本  %s\n" "$(date '+%H:%M:%S')" "$s" "$rc" \
           "$(( $(date +%s) - t0 ))" "$n" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    if [ "$rc" -ne 0 ] || [ "$n" -ne 56 ]; then n_ng=$((n_ng+1)); else n_ok=$((n_ok+1)); fi
done

echo "===== 完了 $(date '+%F %T') ====="
echo "成功 ${n_ok} / スキップ ${n_skip} / 失敗 ${n_ng}"
echo "合計 $(du -sh "$OUT" 2>/dev/null | cut -f1)"
[ "$n_ng" -eq 0 ] && touch logs/export_onnx_30seed.done
exit 0
