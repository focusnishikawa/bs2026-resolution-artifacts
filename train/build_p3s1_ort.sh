#!/usr/bin/env bash
# DINOv2-L の p3 のうち、FP16 で壊れる後半 (p3s1 = blocks 21-23 + norm + head) だけを
# パート単位の ORT 最適化版でビルドし直す.
#
# ここまでに分かったこと:
#   - p3 全体を FP16 でビルドすると必ず定数エンジンになる (4 回再現)
#   - p3 を 2 分割すると **前半 p3s0 (blocks 18-20) は FP16 で正常** (64/64 が相異なる出力)、
#     後半 p3s1 が壊れている
#   - p3 全体では ORT 最適化版が正しく動いた (ただし FP32 になり 7.50 ms)
#
# そこで壊れている p3s1 にだけ ORT 最適化を掛ける。うまくいけば
#   p3s0 (1.81 ms, FP16) + p3s1_ort
# となり、FP32 の 7.50 ms より速くなる可能性がある。
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
rm -f "$E/logs/p3s1ort.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }
msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }

eng="$E/engines_split/dinov2_l_r32_p3s1_ort.engine"
log="$E/logs/build_p3s1_ort.log"
rm -f "$eng"
t0=$(date +%s)
"$TRTEXEC" --onnx="$E/onnx_split_p3/dinov2_l_r32_p3s1_ort.onnx" --saveEngine="$eng" \
    --fp16 --workspace=1024 --warmUp=2000 --iterations=300 --avgRuns=100 > "$log" 2>&1
echo "build rc=$? $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"

for v in mid half; do
    "$E/orin_infer_chain" "$E/preds_orin/p3split_ort_${v}.csv" 64 \
        "$E/inputs/dinov2_l_r32_p012_${v}.fp16.bin" \
        "$E/engines_split/dinov2_l_r32_p3s0.engine" "$eng" \
        > "$E/logs/infer_p3split_ort_${v}.log" 2>&1
    echo "$v: 相異なる出力=$(tail -n +2 "$E/preds_orin/p3split_ort_${v}.csv" | cut -d, -f2- | sort -u | wc -l)/64  先頭=$(sed -n 2p "$E/preds_orin/p3split_ort_${v}.csv" | cut -d, -f2-4)"
done

# 正しく動いていれば全数 5 段チェーンも通す
d=$(tail -n +2 "$E/preds_orin/p3split_ort_mid.csv" | cut -d, -f2- | sort -u | wc -l)
if [ "$d" -gt 1 ]; then
    "$E/orin_infer_chain" "$E/preds_orin/dinov2_l_r32_split5opt_FP16.csv" 1882 \
        "$E/inputs/real_dinov2_l_r32.fp16.bin" \
        "$E/engines_split/dinov2_l_r32_p0.engine" "$E/engines_split/dinov2_l_r32_p1.engine" \
        "$E/engines_split/dinov2_l_r32_p2.engine" \
        "$E/engines_split/dinov2_l_r32_p3s0.engine" "$eng" \
        > "$E/logs/infer_dinov2_split5opt.log" 2>&1
    echo "全数 5 段: rc=$? 相異なる出力=$(tail -n +2 "$E/preds_orin/dinov2_l_r32_split5opt_FP16.csv" | cut -d, -f2- | sort -u | wc -l)/1882"
fi
touch "$E/logs/p3s1ort.done"
