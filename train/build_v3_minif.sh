#!/usr/bin/env bash
# 「If だけを畳んだ」DINOv3-L を分割・ビルドし、ORT 全体最適化版と latency を比べる.
#
# 検証したいこと: DINOv3-L の 1 段が DINOv2-L より 1.4 倍重い (8.21 対 5.82 ms/段) のは、
#   `If` を消すためにグラフ全体を onnxruntime の基本最適化へ通しており、その結果
#   TensorRT の融合が効かなくなっているからではないか。
#
# 比較する 2 系列は **重み・分割点・精度構成がすべて同一**で、違いは準備経路だけ:
#   (a) 既存 = ORT 全体最適化 (ノード 3,600 -> 2,714)          … engines_split/dinov3_l_r<N>_p<k>_fp32.engine
#   (b) 新規 = If だけ畳む     (ノード 3,600 -> 3,601)          … engines_split/dinov3_l_r<N>_minif_p<k>.engine
#
# 前段の `fold_if_minimal.py` は HPC 側で実行済みである必要がある
# (onnx/dinov3_l_r<N>_fp16_minif.onnx が存在すること)。
#
# 起動 (Orin): nohup setsid bash build_v3_minif.sh <res...> > logs/v3_minif_master.log 2>&1 < /dev/null &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
MEASOPT="--warmUp=2000 --iterations=200 --avgRuns=50"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
RESLIST="${*:-32}"
rm -f "$E/logs/v3_minif.done"

n=$(pgrep -cx trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }
median3() { printf '%s\n' "$@" | sort -g | sed -n 2p; }

meas3() {
    local eng="$1" pfx="$2" v=() rep m
    : > "${pfx}.log"
    for rep in 1 2 3; do
        "$TRTEXEC" --loadEngine="$eng" $MEASOPT >> "${pfx}.log" 2>&1
        v+=("$(grep -aoE 'GPU Compute Time:.*mean = [0-9.]+' "${pfx}.log" | tail -1 | grep -oE '[0-9.]+$')")
    done
    m=$(median3 "${v[@]}")
    echo "$m" > "${pfx}.median"
    echo "$m"
}

echo "[$(date '+%m/%d %H:%M:%S')] ##### DINOv3-L: If だけ畳んだ版をビルド #####"

for R in $RESLIST; do
    src="$E/onnx_split_minif/dinov3_l_r${R}_fp16_minif"
    echo "[$(date '+%H:%M:%S')] ===== N=${R} ====="
    total=0; ok=1
    for k in 0 1 2 3; do
        onnx="${src}_p${k}.onnx"
        eng="$E/engines_split/dinov3_l_r${R}_minif_p${k}.engine"
        log="$E/logs/build_v3minif_r${R}_p${k}.log"
        if [ -f "$eng" ]; then
            echo "  [skip] p${k} $(msize "$eng") MB"
        elif [ ! -f "$onnx" ]; then
            echo "  [NG] ONNX なし $(basename "$onnx")"; ok=0; break
        else
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT $FP32 > "$log" 2>&1
            rc=$?
            echo "  p${k} rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB"
            [ $rc -ne 0 ] && { grep -aE '\[E\]|Killed' "$log" | tail -3; ok=0; break; }
        fi
        m=$(meas3 "$eng" "$E/logs/meas_v3minif_r${R}_p${k}")
        echo "     -> ${m} ms"
        total=$(awk -v a="$total" -v b="$m" 'BEGIN{printf "%.3f", a+b}')
    done
    if [ $ok -eq 1 ]; then
        echo "  ★ N=${R} If だけ畳んだ版 4 段合計 = ${total} ms"
        # 既存 (ORT 全体最適化版) と比べる
        prev=0
        for k in 0 1 2 3; do
            f="$E/logs/meas_v3_r${R}_p${k}.median"
            [ -f "$f" ] && prev=$(awk -v a="$prev" -v b="$(cat "$f")" 'BEGIN{printf "%.3f", a+b}')
        done
        [ "$prev" != "0" ] && awk -v a="$total" -v b="$prev" \
            'BEGIN{printf "  ★ 比較: If のみ %.3f ms 対 ORT 全体最適化 %.3f ms (%.2f 倍)\n", a, b, a/b}'
    fi
done

touch "$E/logs/v3_minif.done"
echo "[$(date '+%m/%d %H:%M:%S')] ##### 完了 #####"
