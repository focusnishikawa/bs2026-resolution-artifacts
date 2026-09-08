#!/usr/bin/env bash
# DINOv3-L の p3 も FP32 でビルドし、全 4 段 FP32 のチェーンを完成させる.
#
# 経緯: p0-p2 だけを FP32 にしたところ N=64 は正常化したが、N=128/224 は定数のままだった。
# p3 は出力 (logits) こそ小さいが、内部には他の段と同じ **1.55e5 の残差ストリーム**が流れる。
# fp16 カーネルが選ばれた瞬間に inf へ飽和するので、p3 も FP32 にする必要がある。
# (N=64 で p3 が fp16 のまま通ったのは、TRT がたまたま該当層を FP32 で組んだだけ)
#
# usage: bash build_v3_p3_fp32.sh <res...>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
rm -f "$E/logs/build_v3_p3_fp32.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }
msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }

for R in "$@"; do
    base="dinov3_l_r${R}_fp16_sim"
    [ -f "$E/onnx_split/${base}_p3_ort.onnx" ] || base="dinov3_l_r${R}"
    eng="$E/engines_split/dinov3_l_r${R}_p3_fp32.engine"
    log="$E/logs/build_v3_r${R}_p3_fp32.log"
    echo "[$(date '+%H:%M:%S')] ===== N=${R} の p3 を FP32 でビルド ====="
    if [ ! -f "$eng" ]; then
        t0=$(date +%s)
        "$TRTEXEC" --onnx="$E/onnx_split/${base}_p3_ort.onnx" --saveEngine="$eng" \
            --fp16 --precisionConstraints=obey --layerPrecisions='*':fp32 --workspace=1024 \
            --warmUp=2000 --iterations=200 --avgRuns=50 > "$log" 2>&1
        rc=$?
        echo "  p3 (FP32) rc=$rc $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
        [ $rc -ne 0 ] && { grep -aE "\[E\]|Killed" "$log" | tail -2; continue; }
    else
        echo "  [skip] engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
    fi

    total=0
    for k in 0 1 2; do
        total=$(awk -v a="$total" -v b="$(mean_ms "$E/logs/build_v3_r${R}_p${k}_fp32.log")" 'BEGIN{printf "%.3f", a+b}')
    done
    total=$(awk -v a="$total" -v b="$(mean_ms "$log")" 'BEGIN{printf "%.3f", a+b}')
    echo "  ★ N=${R} の 4 段合計 (全段 FP32) = ${total} ms"

    "$E/orin_infer_chain" "$E/preds_orin/dinov3_l_r${R}_fp32all64.csv" 64 \
        "$E/inputs/real_dinov3_l_r${R}.fp16.bin" \
        "$E/engines_split/dinov3_l_r${R}_p0_fp32.engine" "$E/engines_split/dinov3_l_r${R}_p1_fp32.engine" \
        "$E/engines_split/dinov3_l_r${R}_p2_fp32.engine" "$eng" \
        > "$E/logs/infer_v3_r${R}_fp32all64.log" 2>&1
    d=$(tail -n +2 "$E/preds_orin/dinov3_l_r${R}_fp32all64.csv" | cut -d, -f2- | sort -u | wc -l)
    echo "  入力依存性: 相異なる出力=${d}/64 $([ "$d" -le 1 ] && echo '<< まだ定数')"
done
touch "$E/logs/build_v3_p3_fp32.done"
echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
