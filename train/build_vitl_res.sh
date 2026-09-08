#!/usr/bin/env bash
# ViT-L (DINOv2-L) を N=64/128/224 でも Orin にビルドして latency を測る.
#
# 目的: 報告書 4.4 の限界「ViT-L の Orin 実測は N=32 のみ」を埋め、
#       3.3 の結論「解像度を削っても速くならない」が ViT-L の分割チェーンでも成り立つか見る。
#
# N=32 で確立した最終構成をそのまま適用する:
#   p0 / p1 / p2 / p3s0 は FP16、**p3s1 (最後のブロック群 + norm + head) だけ FP32 強制**。
#   FP16 のままだと p3s1 が入力に依存しない定数を返すため (4 回再現)。
#
# 速度が出ても採用しない。**入力を変えて出力が変わるか**を必ず確かめる。
#
# usage: bash build_vitl_res.sh <res...>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split
SP3=$E/onnx_split_p3
RESLIST="${*:-64 128 224}"
rm -f "$E/logs/build_vitl_res.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }
msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }

for R in $RESLIST; do
    base="dinov2_l_r${R}_fp16_sim"
    echo "[$(date '+%H:%M:%S')] ===== DINOv2-L N=${R} ====="
    total=0
    ok=1
    for part in p0 p1 p2; do
        onnx="$S/${base}_${part}.onnx"
        eng="$E/engines_split/dinov2_l_r${R}_${part}.engine"
        log="$E/logs/build_v2_r${R}_${part}.log"
        if [ -f "$eng" ]; then
            ms=$(mean_ms "$log")
            echo "  [skip] ${part} engine=$(msize "$eng") MB mean=${ms} ms"
        else
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 --workspace=1024 \
                --warmUp=2000 --iterations=200 --avgRuns=50 > "$log" 2>&1
            rc=$?
            ms=$(mean_ms "$log")
            echo "  ${part} rc=$rc $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=${ms} ms"
            [ $rc -ne 0 ] && { ok=0; grep -aE "\[E\]|Killed" "$log" | tail -2; break; }
        fi
        total=$(awk -v a="$total" -v b="${ms:-0}" 'BEGIN{printf "%.3f", a+b}')
    done
    [ $ok -eq 0 ] && { echo "  [NG] N=${R} は途中で失敗"; continue; }

    # p3 前半は FP16、後半は FP32 強制
    for k in 0 1; do
        onnx="$SP3/${base}_p3_p${k}.onnx"
        eng="$E/engines_split/dinov2_l_r${R}_p3s${k}.engine"
        log="$E/logs/build_v2_r${R}_p3s${k}.log"
        opts="--fp16"
        [ "$k" = "1" ] && opts="--fp16 --precisionConstraints=obey --layerPrecisions=*:fp32"
        if [ ! -f "$eng" ]; then
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" $opts --workspace=1024 \
                --warmUp=2000 --iterations=200 --avgRuns=50 > "$log" 2>&1
            rc=$?
            echo "  p3s${k} rc=$rc $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms $([ "$k" = "1" ] && echo '(FP32 強制)')"
            [ $rc -ne 0 ] && { ok=0; grep -aE "\[E\]|Killed" "$log" | tail -2; break; }
        else
            echo "  [skip] p3s${k} engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
        fi
        total=$(awk -v a="$total" -v b="$(mean_ms "$log")" 'BEGIN{printf "%.3f", a+b}')
    done
    [ $ok -eq 0 ] && continue
    echo "  ★ N=${R} の 5 段合計 = ${total} ms"

    # 入力依存性 (実画像 64 枚。定数エンジンを掴んでいないか)
    bin="$E/inputs/real_dinov2_l_r${R}.fp16.bin"
    if [ -f "$bin" ]; then
        "$E/orin_infer_chain" "$E/preds_orin/dinov2_l_r${R}_chain64.csv" 64 "$bin" \
            "$E/engines_split/dinov2_l_r${R}_p0.engine" "$E/engines_split/dinov2_l_r${R}_p1.engine" \
            "$E/engines_split/dinov2_l_r${R}_p2.engine" \
            "$E/engines_split/dinov2_l_r${R}_p3s0.engine" "$E/engines_split/dinov2_l_r${R}_p3s1.engine" \
            > "$E/logs/infer_v2_r${R}_chain64.log" 2>&1
        d=$(tail -n +2 "$E/preds_orin/dinov2_l_r${R}_chain64.csv" | cut -d, -f2- | sort -u | wc -l)
        echo "  入力依存性: 相異なる出力=${d}/64 $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"
    fi
done
echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
touch "$E/logs/build_vitl_res.done"
