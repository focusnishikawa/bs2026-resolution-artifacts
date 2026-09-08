#!/usr/bin/env bash
# DINOv2-L の p3 後半 (blocks 21-23 + norm + head) だけを FP32 カーネルでビルドする.
#
# ここまでの切り分け:
#   - p3 全体を FP16 でビルド -> 定数エンジン (4 回再現: 元 / static / ORT / 再ビルド)
#   - p3 を 2 分割すると前半 p3s0 は FP16 で正常 (1.81 ms)、後半 p3s1 が壊れる
#   - p3s1 は ORT 最適化版でも FP16 では壊れたまま (1.87 ms で定数)
#   - 正しく動いた唯一の構成は、TRT がたまたま FP32 カーネルを選んだ p3 全体の ORT 版 (7.50 ms)
#
# つまり **この区間は FP32 でなければ正しい結果を返さない**。ならば FP32 が要るのは
# 後半だけのはずなので、p3s1 だけ FP32 を強制すれば
#   p3s0 (FP16 1.81 ms) + p3s1 (FP32) < p3 全体を FP32 でやる 7.50 ms
# となり全体が縮む見込み。
#
# ONNX 自体が fp16 なので --fp16 は外せない (外すとパースが Error Code 4 で落ちる)。
# そこで --fp16 は付けたまま precisionConstraints で全層 FP32 を指示する。
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
rm -f "$E/logs/p3s1fp32.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }
msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }

eng="$E/engines_split/dinov2_l_r32_p3s1_fp32.engine"
log="$E/logs/build_p3s1_fp32.log"
rm -f "$eng"
t0=$(date +%s)
"$TRTEXEC" --onnx="$E/onnx_split_p3/dinov2_l_r32_p3_static_p1.onnx" --saveEngine="$eng" \
    --fp16 --precisionConstraints=obey --layerPrecisions='*':fp32 \
    --workspace=1024 --warmUp=2000 --iterations=300 --avgRuns=100 > "$log" 2>&1
rc=$?
echo "build rc=$rc $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
if [ $rc -ne 0 ]; then
    grep -aE "\[E\]" "$log" | head -3
    touch "$E/logs/p3s1fp32.done"; exit 1
fi

for v in mid half; do
    "$E/orin_infer_chain" "$E/preds_orin/p3split_fp32_${v}.csv" 64 \
        "$E/inputs/dinov2_l_r32_p012_${v}.fp16.bin" \
        "$E/engines_split/dinov2_l_r32_p3s0.engine" "$eng" \
        > "$E/logs/infer_p3split_fp32_${v}.log" 2>&1
    echo "$v: 相異なる出力=$(tail -n +2 "$E/preds_orin/p3split_fp32_${v}.csv" | cut -d, -f2- | sort -u | wc -l)/64"
done

d=$(tail -n +2 "$E/preds_orin/p3split_fp32_mid.csv" | cut -d, -f2- | sort -u | wc -l)
if [ "$d" -gt 1 ]; then
    "$E/orin_infer_chain" "$E/preds_orin/dinov2_l_r32_split5fp32_FP16.csv" 1882 \
        "$E/inputs/real_dinov2_l_r32.fp16.bin" \
        "$E/engines_split/dinov2_l_r32_p0.engine" "$E/engines_split/dinov2_l_r32_p1.engine" \
        "$E/engines_split/dinov2_l_r32_p2.engine" \
        "$E/engines_split/dinov2_l_r32_p3s0.engine" "$eng" \
        > "$E/logs/infer_dinov2_split5fp32.log" 2>&1
    echo "全数 5 段: rc=$? 相異なる出力=$(tail -n +2 "$E/preds_orin/dinov2_l_r32_split5fp32_FP16.csv" | cut -d, -f2- | sort -u | wc -l)/1882"
fi
touch "$E/logs/p3s1fp32.done"
