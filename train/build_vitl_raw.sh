#!/usr/bin/env bash
# DINOv2-L を「生の分割」(ORT 最適化なし) でビルド・全数推論・3 回測定まで一気に行う.
#
# なぜ作り直すか: 全 14 水準の実測で DINOv2-L だけ N=32 が 16.4 ms と両隣 (N=16 40.3 / N=48 41.0)
# から外れた。N=32 のエンジンだけ ORT 最適化前の生の ONNX から分割していたためで、
# 解像度の効果ではなく**準備経路の差**だった。全水準を N=32 と同じ経路へ揃える。
#
# 精度構成は従来どおり p3s1 (最終ブロック群+head) のみ FP32、他は FP16。
# 速度が出ても採用しない。入力を変えて出力が変わることを毎水準で確認する。
#
# usage: nohup setsid bash build_vitl_raw.sh <res...> > logs/build_vitl_raw_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split_raw
SP3=$E/onnx_split_raw_p3
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
MEASOPT="--warmUp=2000 --iterations=200 --avgRuns=50"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
NTEST=1882
rm -f "$E/logs/build_vitl_raw.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }
median3() { printf '%s\n' "$@" | sort -g | sed -n 2p; }

for R in "$@"; do
    base="dinov2_l_r${R}_fp16"
    echo "[$(date '+%H:%M:%S')] ===== DINOv2-L N=${R} (生の分割) ====="
    ok=1
    for part in p0 p1 p2 p3s0 p3s1; do
        case "$part" in
            p3s0) onnx="$SP3/${base}_p3_p0.onnx"; opts="" ;;
            p3s1) onnx="$SP3/${base}_p3_p1.onnx"; opts="$FP32" ;;
            *)    onnx="$S/${base}_${part}.onnx";  opts="" ;;
        esac
        eng="$E/engines_split/dinov2_l_r${R}_raw_${part}.engine"
        log="$E/logs/build_v2raw_r${R}_${part}.log"
        if [ -f "$eng" ]; then
            echo "  [skip] ${part} $(msize "$eng") MB"
        elif [ ! -f "$onnx" ]; then
            echo "  [NG] ONNX なし $(basename "$onnx")"; ok=0; break
        else
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT $opts > "$log" 2>&1
            rc=$?
            echo "  ${part} rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB$([ -n "$opts" ] && echo ' (FP32 強制)')"
            [ $rc -ne 0 ] && { grep -aE "\[E\]|Killed" "$log" | tail -2; ok=0; break; }
        fi
    done
    [ $ok -eq 0 ] && { echo "  [NG] N=${R} は途中で失敗"; continue; }

    G=$E/engines_split
    CH="$G/dinov2_l_r${R}_raw_p0.engine $G/dinov2_l_r${R}_raw_p1.engine $G/dinov2_l_r${R}_raw_p2.engine \
        $G/dinov2_l_r${R}_raw_p3s0.engine $G/dinov2_l_r${R}_raw_p3s1.engine"

    # 全数推論 (精度・argmax 一致率・入力依存性をまとめて確認)
    csv="$E/preds_orin/dinov2_l_r${R}_raw_full.csv"
    if [ ! -s "$csv" ] || [ "$(wc -l < "$csv")" -ne $((NTEST + 1)) ]; then
        t0=$(date +%s)
        "$E/orin_infer_chain" "$csv" "$NTEST" "$E/inputs/real_dinov2_l_r${R}.fp16.bin" $CH \
            > "${csv%.csv}.log" 2>&1
        echo "  推論 $(( $(date +%s) - t0 ))s"
    fi
    d=$(tail -n +2 "$csv" | cut -d, -f2- | sort -u | wc -l)
    echo "  入力依存性: 相異なる出力=${d}/${NTEST} $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"

    # 3 回測定 (他構成と手順を揃える)
    total=0
    for part in p0 p1 p2 p3s0 p3s1; do
        eng="$G/dinov2_l_r${R}_raw_${part}.engine"
        log="$E/logs/meas_v2raw_r${R}_${part}.log"
        : > "$log"; v=()
        for rep in 1 2 3; do
            "$TRTEXEC" --loadEngine="$eng" $MEASOPT >> "$log" 2>&1
            v+=("$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")")
        done
        m=$(median3 "${v[@]}")
        echo "$m" > "${log%.log}.median"
        echo "  ${part}: ${v[*]} -> 中央値 ${m} ms"
        total=$(awk -v a="$total" -v b="$m" 'BEGIN{printf "%.3f", a+b}')
    done
    echo "  ★ N=${R} の 5 段合計 = ${total} ms"
done
touch "$E/logs/build_vitl_raw.done"
echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
