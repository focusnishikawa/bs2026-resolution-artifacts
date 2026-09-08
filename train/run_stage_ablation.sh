#!/usr/bin/env bash
# 「DINOv2-L 3.80 と DINOv3-L 2.76 の比の差」の残差要因を切り分ける実験.
#
# 予備分析 (既存データ) でここまで分かっている:
#   ・1 トークンあたりの計算コストは両モデルでほぼ同じ (N=224 で 0.431 対 0.451 ms/token)
#   ・差は「到達トークン数が 1.28 倍 (257 対 201)」と「段あたり固定費が 0.71 倍」の積で
#     ちょうど説明できる (1.22 / 0.885 = 1.38 = 3.80/2.76)
#   ・残る問いは **なぜ DINOv3-L の 1 段が重いのか** (5.82 対 8.21 ms/段)
#
# 仮説: DINOv3-L は `If` を畳むためにグラフ全体を onnxruntime の基本最適化に通しており、
#       そのせいで TensorRT の融合が効かず 1 段が重い (準備経路の知見と同じ機序)。
#       DINOv2-L は最適化を通していないので融合が残り、1 段が軽い。
#
# 実験1 (段数の寄与): DINOv2-L を **4 段**で全段 FP32 ビルドする (配備構成は 5 段)。
#   重み・分割点の前半・精度構成は不変で、p3 を分割しないだけ。
#   段数だけが減るので、`段数 x 段あたり固定費` というモデルが正しいかを検証できる。
#   4 段の合計が 5 段より約 1 段ぶん軽くなれば固定費モデルは支持される。
#
# 実験4 (分割そのものの代償): DINOv3-L を **単体エンジン**でビルドし、4 段分割と比べる。
#   分割しない場合の固定費が分かるので、段あたり固定費を直接見積もれる。
#   ⚠️ DINOv2-L は単体ではビルドできない (全 2,430 層が単一 MYELIN ノードに融合され OOM)。
#
# 起動 (Orin): nohup setsid bash run_stage_ablation.sh > logs/stage_ablation_master.log 2>&1 < /dev/null &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split_raw
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
MEASOPT="--warmUp=2000 --iterations=200 --avgRuns=50"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
RES4="16 64 112 128 224"          # 実験1 (N=32 は既に 4 段で作成済み)
RES_SINGLE="16 32 64 112 128 224" # 実験4
rm -f "$E/logs/stage_ablation.done"

n=$(pgrep -cx trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }
median3() { printf '%s\n' "$@" | sort -g | sed -n 2p; }

# meas3 <engine> <logprefix> -> 中央値を .median へ
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

echo "[$(date '+%m/%d %H:%M:%S')] ##### 実験1: DINOv2-L を 4 段で全段 FP32 #####"
for R in $RES4; do
    echo "[$(date '+%H:%M:%S')] --- N=${R} (4 段) ---"
    ok=1; total=0
    for k in 0 1 2 3; do
        onnx="$S/dinov2_l_r${R}_fp16_p${k}.onnx"
        eng="$E/engines_split/dinov2_l_r${R}_f32s4_p${k}.engine"
        log="$E/logs/build_v2f32s4_r${R}_p${k}.log"
        if [ -f "$eng" ]; then
            echo "  [skip] p${k} $(msize "$eng") MB"
        elif [ ! -f "$onnx" ]; then
            echo "  [NG] ONNX なし $(basename "$onnx")"; ok=0; break
        else
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT $FP32 > "$log" 2>&1
            rc=$?
            echo "  p${k} rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB"
            [ $rc -ne 0 ] && { grep -aE '\[E\]|Killed' "$log" | tail -2; ok=0; break; }
        fi
        m=$(meas3 "$eng" "$E/logs/meas_v2f32s4_r${R}_p${k}")
        total=$(awk -v a="$total" -v b="$m" 'BEGIN{printf "%.3f", a+b}')
    done
    [ $ok -eq 1 ] && echo "  ★ N=${R} 4 段合計 = ${total} ms"
done

echo "[$(date '+%m/%d %H:%M:%S')] ##### 実験4: DINOv3-L を単体エンジンで #####"
for R in $RES_SINGLE; do
    onnx="$E/onnx/dinov3_l_r${R}_fp16_sim.onnx"
    eng="$E/engines_split/dinov3_l_r${R}_single_f32.engine"
    log="$E/logs/build_v3single_r${R}.log"
    if [ ! -f "$onnx" ]; then echo "  [miss] N=${R} の ONNX なし"; continue; fi
    if [ -f "$eng" ]; then
        echo "  [skip] N=${R} $(msize "$eng") MB"
    else
        t0=$(date +%s)
        "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT $FP32 > "$log" 2>&1
        rc=$?
        echo "  N=${R} rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB"
        # 単体ビルドはメモリを食うので失敗しうる。失敗しても次の水準へ進む
        [ $rc -ne 0 ] && { grep -aE '\[E\]|Killed' "$log" | tail -2; continue; }
    fi
    m=$(meas3 "$eng" "$E/logs/meas_v3single_r${R}")
    echo "  ★ N=${R} 単体 = ${m} ms"
done

touch "$E/logs/stage_ablation.done"
echo "[$(date '+%m/%d %H:%M:%S')] ##### 完了 #####"
