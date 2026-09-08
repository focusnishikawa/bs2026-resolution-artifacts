#!/usr/bin/env bash
# DINOv2-L を **全段 FP32** でビルドし、混合精度の配備構成との対照を取る.
#
# 目的: 図 6 (argmax 一致率) で、DINOv2-L の一致率低下が
#         (a) 半精度化によるもの        -> 全段 FP32 にすれば一致率が上がるはず
#         (b) 低解像度そのものによるもの -> 全段 FP32 にしても上がらないはず
#       のどちらなのかを切り分ける。配備構成 (p3s1 のみ FP32) は両者が混ざっていて判別できない。
#
# 構成: 生の分割 ONNX (配備構成と同じ経路) の 5 パートすべてに FP32 を強制する。
#       重み・分割点は配備構成と完全に同一で、**変えるのは演算精度だけ**。
#
# 水準: 図 6 に載せる 6 点のみ (16/32/64/112/128/224)。
#       DINOv2-L は patch14 なので実入力は 14/28/70/112/126/224 になる。
#
# usage (Orin): nohup setsid bash build_v2_fullfp32.sh > logs/v2_fullfp32_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split_raw
SP3=$E/onnx_split_raw_p3
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
MEASOPT="--warmUp=2000 --iterations=200 --avgRuns=50"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
NTEST=1882
RESLIST="${*:-16 32 64 112 128 224}"
rm -f "$E/logs/v2_fullfp32.done"

n=$(pgrep -cx trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }
median3() { printf '%s\n' "$@" | sort -g | sed -n 2p; }

echo "[$(date '+%m/%d %H:%M:%S')] ===== DINOv2-L 全段 FP32 (水準: $RESLIST) ====="

for R in $RESLIST; do
    base="dinov2_l_r${R}_fp16"
    echo "[$(date '+%H:%M:%S')] ===== N=${R} ====="
    ok=1
    for part in p0 p1 p2 p3s0 p3s1; do
        case "$part" in
            p3s0) onnx="$SP3/${base}_p3_p0.onnx" ;;
            p3s1) onnx="$SP3/${base}_p3_p1.onnx" ;;
            *)    onnx="$S/${base}_${part}.onnx" ;;
        esac
        eng="$E/engines_split/dinov2_l_r${R}_f32_${part}.engine"
        log="$E/logs/build_v2f32_r${R}_${part}.log"
        if [ -f "$eng" ]; then
            echo "  [skip] ${part} $(msize "$eng") MB"
        elif [ ! -f "$onnx" ]; then
            echo "  [NG] ONNX なし $(basename "$onnx")"; ok=0; break
        else
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT $FP32 > "$log" 2>&1
            rc=$?
            echo "  ${part} rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB (全段FP32)"
            [ $rc -ne 0 ] && { grep -aE "\[E\]|Killed" "$log" | tail -2; ok=0; break; }
        fi
    done
    [ $ok -eq 0 ] && { echo "  [NG] N=${R} は途中で失敗"; continue; }

    G=$E/engines_split
    CH="$G/dinov2_l_r${R}_f32_p0.engine $G/dinov2_l_r${R}_f32_p1.engine $G/dinov2_l_r${R}_f32_p2.engine \
        $G/dinov2_l_r${R}_f32_p3s0.engine $G/dinov2_l_r${R}_f32_p3s1.engine"

    # 全数推論 (一致率を出すのが本題)
    csv="$E/preds_orin/dinov2_l_r${R}_f32_full.csv"
    if [ ! -s "$csv" ] || [ "$(wc -l < "$csv")" -ne $((NTEST + 1)) ]; then
        t0=$(date +%s)
        "$E/orin_infer_chain" "$csv" "$NTEST" "$E/inputs/real_dinov2_l_r${R}.fp16.bin" $CH \
            > "${csv%.csv}.log" 2>&1
        echo "  推論 $(( $(date +%s) - t0 ))s"
    fi
    # **速度もサイズも正常性の証拠にならない**ので、入力依存性で必ず確かめる
    d=$(tail -n +2 "$csv" | cut -d, -f2- | sort -u | wc -l)
    echo "  入力依存性: 相異なる出力=${d}/${NTEST} $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"

    # latency も配備構成と比べられるよう 3 回測定しておく (30 回版は measure_all30.sh 側)
    total=0
    for part in p0 p1 p2 p3s0 p3s1; do
        eng="$G/dinov2_l_r${R}_f32_${part}.engine"
        log="$E/logs/meas_v2f32_r${R}_${part}.log"
        : > "$log"; v=()
        for rep in 1 2 3; do
            "$TRTEXEC" --loadEngine="$eng" $MEASOPT >> "$log" 2>&1
            v+=("$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")")
        done
        m=$(median3 "${v[@]}")
        echo "$m" > "${log%.log}.median"
        total=$(awk -v a="$total" -v b="$m" 'BEGIN{printf "%.3f", a+b}')
    done
    echo "  ★ N=${R} 全段FP32 の 5 段合計 = ${total} ms"
done

touch "$E/logs/v2_fullfp32.done"
echo "[$(date '+%m/%d %H:%M:%S')] ===== 完了 ====="
