#!/usr/bin/env bash
# DINOv3-L の分割チェーンを FP32 でビルドし直す (fp16 レンジ超過の回避).
#
# 判明した原因: DINOv3-L の残差ストリームは **最大 1.55e5** に達し、fp16 の上限 65,504 を
# 大きく超える (ORT で全解像度を確認。DINOv2-L は 1〜9 で余裕がある)。
# fp16 カーネルを選ばれた段は inf に飽和し、以降が入力に依存しない定数になる。
# N=32 で 4 段とも動いていたのは、TRT がたまたま該当層を FP32 で組んだためにすぎない。
#
# よって p0-p2 (巨大な活性値を運ぶ段) は FP32 を強制する。p3 は出力が logits で値が小さいので
# fp16 のままでよい (段別診断でも p3 単体は健全だった)。
#
# usage: bash build_v3_fp32.sh <res>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
R="${1:-64}"
# 分割 ONNX の名前は解像度によって 2 通りある (r32 は fp16_sim を含まない)
base="dinov3_l_r${R}_fp16_sim"
[ -f "$E/onnx_split/${base}_p0_ort.onnx" ] || base="dinov3_l_r${R}"
rm -f "$E/logs/build_v3_fp32_r${R}.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }
msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }

echo "[$(date '+%H:%M:%S')] ===== DINOv3-L N=${R} を FP32 でビルド ====="
total=0
for k in 0 1 2 3; do
    onnx="$E/onnx_split/${base}_p${k}_ort.onnx"
    if [ "$k" -le 2 ]; then
        eng="$E/engines_split/dinov3_l_r${R}_p${k}_fp32.engine"
        log="$E/logs/build_v3_r${R}_p${k}_fp32.log"
        opts="--fp16 --precisionConstraints=obey --layerPrecisions=*:fp32"
        tagname="p${k} (FP32 強制)"
    else
        eng="$E/engines_split/dinov3_l_r${R}_p${k}_ort.engine"   # 既存の fp16 版を使う
        log="$E/logs/build_v3_r${R}_p${k}.log"
        opts=""
        tagname="p${k} (既存 FP16)"
    fi
    if [ ! -f "$eng" ]; then
        t0=$(date +%s)
        "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" $opts --workspace=1024 \
            --warmUp=2000 --iterations=200 --avgRuns=50 > "$log" 2>&1
        rc=$?
        echo "  ${tagname} rc=$rc $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
        [ $rc -ne 0 ] && { grep -aE "\[E\]|Killed" "$log" | tail -2; exit 1; }
    else
        echo "  [skip] ${tagname} engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
    fi
    total=$(awk -v a="$total" -v b="$(mean_ms "$log")" 'BEGIN{printf "%.3f", a+b}')
done
echo "  ★ N=${R} の 4 段合計 (p0-p2 FP32) = ${total} ms"

"$E/orin_infer_chain" "$E/preds_orin/dinov3_l_r${R}_fp32chain64.csv" 64 \
    "$E/inputs/real_dinov3_l_r${R}.fp16.bin" \
    "$E/engines_split/dinov3_l_r${R}_p0_fp32.engine" "$E/engines_split/dinov3_l_r${R}_p1_fp32.engine" \
    "$E/engines_split/dinov3_l_r${R}_p2_fp32.engine" "$E/engines_split/dinov3_l_r${R}_p3_ort.engine" \
    > "$E/logs/infer_v3_r${R}_fp32chain64.log" 2>&1
d=$(tail -n +2 "$E/preds_orin/dinov3_l_r${R}_fp32chain64.csv" | cut -d, -f2- | sort -u | wc -l)
echo "  入力依存性: 相異なる出力=${d}/64 $([ "$d" -le 1 ] && echo '<< まだ定数')"
touch "$E/logs/build_v3_fp32_r${R}.done"
echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
