#!/usr/bin/env bash
# ViT-L の 4 分割チェーンをさらに詰める (Orin 側だけで完結する作業).
#
# 現状 (8/19 時点の確定値):
#   DINOv2-L 18.845 ms = 4.03 + 3.66 + 3.66 + 7.50   <- p3_ort だけ 2 倍遅い
#   DINOv3-L 33.579 ms = 7.27 + 12.99 + 6.63 + 6.69  <- p0/p1 が未最適化のまま
#
# 手掛かり: engine サイズを並べると p3_ort だけ 302.7 MB で、他パートの 151.6 MB のちょうど 2 倍。
# ONNX は fp16 (151.3 MB) なのに **engine が FP32 のまま**である。ViT-S/16 の N=64 で
# latency が +39% 外れていたのと同じ「FP16 経路の取りこぼし」で、TRT がビルド時の実測で
# FP32 カーネルを選んでしまったと考えられる。
#
# そこで:
#   (A) DINOv2-L の p3 を作り直す。まず同条件で再ビルドし、それでも FP32 のままなら
#       --precisionConstraints=obey --layerPrecisions=*:fp16 で FP16 を強制する
#   (B) DINOv3-L の p0/p1 もパート単位の ORT 最適化版でビルドする
#       (p2/p3 は同じ処置で 13.6 -> 6.6 ms とほぼ半減した)
#
# 速度だけで採否を決めない。壊れたエンジンは計算が消えるぶん速く出るので、
# **入力を変えて出力が変わるか**を必ず確かめる。
#
# 起動 (Orin):
#   setsid nohup bash optimize_vitl_chain.sh > logs/opt_chain_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
cd "$E" || exit 1
rm -f "$E/logs/opt_chain.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }
mean_ms() {
    grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" \
        | grep -oE "[0-9.]+" | tail -1
}

build() {  # build <出力engine> <onnx> <ログ> [追加オプション...]
    local eng="$1" onnx="$2" log="$3"; shift 3
    rm -f "$eng"
    local t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 --workspace=1024 \
        --warmUp=2000 --iterations=300 --avgRuns=100 "$@" > "$log" 2>&1
    local rc=$?
    echo "    rc=$rc  $(( $(date +%s) - t0 ))s  engine=$(msize "$eng") MB  mean=$(mean_ms "$log") ms"
    return $rc
}

echo "[$(date '+%H:%M:%S')] ===== (A) DINOv2-L の p3 を FP16 で作り直す ====="
echo "  現状: p3_ort engine=$(msize $E/engines_split/dinov2_l_r32_p3_ort.engine) MB (他パートは 151.6 MB)"

echo "  [A-1] 同条件で再ビルド (タクティク選択は実測依存なので結果が変わりうる)"
build "$E/engines_split/dinov2_l_r32_p3_ort2.engine" \
      "$E/onnx_split/dinov2_l_r32_p3_ort.onnx" \
      "$E/logs/build_p3_ort2.log"
sz=$(msize "$E/engines_split/dinov2_l_r32_p3_ort2.engine")

BEST="$E/engines_split/dinov2_l_r32_p3_ort2.engine"
# 150 MB 台なら FP16、300 MB 台なら FP32 のまま
if [ "${sz%%.*}" -gt 200 ]; then
    echo "  [A-2] まだ FP32 のままなので FP16 を強制する"
    build "$E/engines_split/dinov2_l_r32_p3_fp16forced.engine" \
          "$E/onnx_split/dinov2_l_r32_p3_ort.onnx" \
          "$E/logs/build_p3_fp16forced.log" \
          --precisionConstraints=obey --layerPrecisions="*":fp16
    sz2=$(msize "$E/engines_split/dinov2_l_r32_p3_fp16forced.engine")
    [ -n "$sz2" ] && [ "${sz2%%.*}" -lt 200 ] && \
        BEST="$E/engines_split/dinov2_l_r32_p3_fp16forced.engine"
fi
echo "  採用候補: $(basename "$BEST") ($(msize "$BEST") MB)"

echo "  [A-3] 入力依存性の確認 (mid と half で出力が変わるか)"
for v in mid half; do
    "$E/orin_infer_chain" "$E/preds_orin/p3new_${v}.csv" 64 \
        "$E/inputs/dinov2_l_r32_p012_${v}.fp16.bin" "$BEST" \
        > "$E/logs/infer_p3new_${v}.log" 2>&1
    echo "    $v: 相異なる出力=$(tail -n +2 "$E/preds_orin/p3new_${v}.csv" | cut -d, -f2- | sort -u | wc -l)/64  先頭=$(sed -n 2p "$E/preds_orin/p3new_${v}.csv" | cut -d, -f2-4)"
done

echo "[$(date '+%H:%M:%S')] ===== (B) DINOv3-L の p0/p1 を ORT 最適化版でビルド ====="
for p in p0 p1; do
    echo "  [$p]"
    build "$E/engines_split/dinov3_l_r32_${p}_ort.engine" \
          "$E/onnx_split/dinov3_l_r32_${p}_ort.onnx" \
          "$E/logs/build_v3_${p}_ort.log"
done

echo "[$(date '+%H:%M:%S')] ===== 全数チェーンで正しさを確認 ====="
"$E/orin_infer_chain" "$E/preds_orin/dinov2_l_r32_split4opt_FP16.csv" 1882 \
    "$E/inputs/real_dinov2_l_r32.fp16.bin" \
    "$E/engines_split/dinov2_l_r32_p0.engine" "$E/engines_split/dinov2_l_r32_p1.engine" \
    "$E/engines_split/dinov2_l_r32_p2.engine" "$BEST" \
    > "$E/logs/infer_dinov2_split4opt.log" 2>&1
echo "  dinov2_l rc=$? 相異なる出力=$(tail -n +2 "$E/preds_orin/dinov2_l_r32_split4opt_FP16.csv" | cut -d, -f2- | sort -u | wc -l)/1882"

"$E/orin_infer_chain" "$E/preds_orin/dinov3_l_r32_split4opt_FP16.csv" 1882 \
    "$E/inputs/real_dinov3_l_r32.fp16.bin" \
    "$E/engines_split/dinov3_l_r32_p0_ort.engine" "$E/engines_split/dinov3_l_r32_p1_ort.engine" \
    "$E/engines_split/dinov3_l_r32_p2_ort.engine" "$E/engines_split/dinov3_l_r32_p3_ort.engine" \
    > "$E/logs/infer_dinov3_split4opt.log" 2>&1
echo "  dinov3_l rc=$? 相異なる出力=$(tail -n +2 "$E/preds_orin/dinov3_l_r32_split4opt_FP16.csv" | cut -d, -f2- | sort -u | wc -l)/1882"

echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
touch "$E/logs/opt_chain.done"
