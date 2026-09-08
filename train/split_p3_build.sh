#!/usr/bin/env bash
# DINOv2-L の p3 を 2 分割して FP16 で動かせるか試す.
#
# 分かっていること:
#   - p3 を FP16 でビルドすると **必ず定数を返すエンジン**になる (3 回とも再現。
#     元の p3 / static 版 / ORT 版の再ビルド、いずれも engine 151 MB で入力非依存)
#   - FP32 でビルドすると正しく動くが 7.50 ms かかる (他パートは FP16 で 3.66 ms)
#   - つまり現在の 18.845 ms のうち p3 の 7.50 ms は「FP32 でしか正しく動かない」ことの代償
#
# モデル全体が単一 MYELIN ForeignNode に融合されて OOM した時と同じ発想で、
# p3 をさらに 2 つに割れば融合単位が小さくなり FP16 でも壊れなくなる可能性がある。
# 切断点は blocks.20 出口 (part0 = blocks 18-20、part1 = blocks 21-23 + norm + head)。
#
# 速度が出ても**入力を変えて出力が変わるか**を必ず確かめてから採用する。
#
# 起動 (Orin):
#   setsid nohup bash split_p3_build.sh > logs/split_p3_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
S=$E/onnx_split_p3
TRTEXEC=/usr/src/tensorrt/bin/trtexec
rm -f "$E/logs/split_p3.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }
mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }

echo "[$(date '+%H:%M:%S')] ===== p3 の 2 分割をビルド ====="
total=0
for k in 0 1; do
    onnx="$S/dinov2_l_r32_p3_static_p${k}.onnx"
    eng="$E/engines_split/dinov2_l_r32_p3s${k}.engine"
    log="$E/logs/build_p3s${k}.log"
    [ -f "$onnx" ] || { echo "  [miss] $onnx"; exit 1; }
    rm -f "$eng"
    t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 --workspace=1024 \
        --warmUp=2000 --iterations=300 --avgRuns=100 > "$log" 2>&1
    ms=$(mean_ms "$log")
    echo "  p3s$k rc=$? $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=${ms} ms"
    total=$(awk -v a="$total" -v b="${ms:-0}" 'BEGIN{printf "%.3f", a+b}')
done
echo "  p3 の 2 分割合計 = ${total} ms (FP32 単体版は 7.50 ms)"

echo "[$(date '+%H:%M:%S')] ===== 入力依存性の確認 ====="
for v in mid half; do
    "$E/orin_infer_chain" "$E/preds_orin/p3split_${v}.csv" 64 \
        "$E/inputs/dinov2_l_r32_p012_${v}.fp16.bin" \
        "$E/engines_split/dinov2_l_r32_p3s0.engine" "$E/engines_split/dinov2_l_r32_p3s1.engine" \
        > "$E/logs/infer_p3split_${v}.log" 2>&1
    echo "  $v: 相異なる出力=$(tail -n +2 "$E/preds_orin/p3split_${v}.csv" | cut -d, -f2- | sort -u | wc -l)/64  先頭=$(sed -n 2p "$E/preds_orin/p3split_${v}.csv" | cut -d, -f2-4)"
done

echo "[$(date '+%H:%M:%S')] ===== 全数 5 段チェーン ====="
"$E/orin_infer_chain" "$E/preds_orin/dinov2_l_r32_split5_FP16.csv" 1882 \
    "$E/inputs/real_dinov2_l_r32.fp16.bin" \
    "$E/engines_split/dinov2_l_r32_p0.engine" "$E/engines_split/dinov2_l_r32_p1.engine" \
    "$E/engines_split/dinov2_l_r32_p2.engine" \
    "$E/engines_split/dinov2_l_r32_p3s0.engine" "$E/engines_split/dinov2_l_r32_p3s1.engine" \
    > "$E/logs/infer_dinov2_split5.log" 2>&1
echo "  rc=$? 相異なる出力=$(tail -n +2 "$E/preds_orin/dinov2_l_r32_split5_FP16.csv" | cut -d, -f2- | sort -u | wc -l)/1882"

echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
touch "$E/logs/split_p3.done"
