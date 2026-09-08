#!/usr/bin/env bash
# N=32 だけ全段 FP32 を **4 分割**で作る (図 6 の対照の穴埋め).
#
# なぜ別スクリプトか: build_v2_fullfp32.sh は 5 分割 (p0/p1/p2/p3s0/p3s1) を前提にしているが、
#   N=32 は最初期に作った水準で命名が独自 (onnx_split/dinov2_l_r32_p{0..3}.onnx) であり、
#   **p3 を 2 分割した ONNX が存在しない**。そのため 5 分割版は N=32 で失敗する。
#
# 4 分割で問題ない理由: p3 を p3s0/p3s1 に分けたのは「p3s1 だけを FP32 にする」ためだった。
#   全段 FP32 では分ける動機が無く、計算は等価である (分割の等価性は ONNX レベルで
#   ビット一致を検証済み)。図 6 は argmax 一致率の図なので分割点の違いは影響しない。
#   ただし **latency は他水準 (5 分割) と直接比較しない**こと。
#
# 実行タイミング: Orin は排他なので、走行中の ViT-L 30 回測定の完了を待ってから始める。
#   測定中に別プロセスが GPU を使うと latency が汚染される。
#
# 起動 (Orin): nohup setsid bash build_v2_fullfp32_r32.sh > logs/v2f32_r32_master.log 2>&1 < /dev/null &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
NTEST=1882
R=32
rm -f "$E/logs/v2f32_r32.done"

echo "[$(date '+%m/%d %H:%M:%S')] ##### N=32 全段 FP32 (4 分割) #####"

# ---- ViT-L 30 回測定の完了を待つ (排他) ----
if [ ! -f "$E/logs/meas30_vitl.done" ]; then
    echo "[$(date '+%m/%d %H:%M:%S')] ViT-L 30 回測定の完了を待機中 (5 分ごとに確認)"
    waited=0
    while [ ! -f "$E/logs/meas30_vitl.done" ]; do
        sleep 300
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')]   待機 ${waited} 分 / ViT-L $(ls "$E"/logs/meas30_v*.stats 2>/dev/null | wc -l)/126"
        fi
        if [ "$(pgrep -cf '[m]easure_all30.sh vitl' 2>/dev/null || echo 0)" -eq 0 ] && [ "$waited" -gt 10 ]; then
            echo "[$(date '+%H:%M:%S')] ViT-L 測定プロセスが見当たらないので待機を打ち切る"
            break
        fi
    done
fi

n=$(pgrep -cx trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }

ok=1
for k in 0 1 2 3; do
    onnx="$S/dinov2_l_r${R}_p${k}.onnx"
    eng="$E/engines_split/dinov2_l_r${R}_f32_p${k}.engine"
    log="$E/logs/build_v2f32_r${R}_p${k}.log"
    if [ -f "$eng" ]; then
        echo "  [skip] p${k} $(msize "$eng") MB"
    elif [ ! -f "$onnx" ]; then
        echo "  [NG] ONNX なし $(basename "$onnx")"; ok=0; break
    else
        t0=$(date +%s)
        "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT $FP32 > "$log" 2>&1
        rc=$?
        echo "  p${k} rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB (全段FP32)"
        [ $rc -ne 0 ] && { grep -aE "\[E\]|Killed" "$log" | tail -2; ok=0; break; }
    fi
done

if [ $ok -eq 1 ]; then
    G=$E/engines_split
    CH="$G/dinov2_l_r${R}_f32_p0.engine $G/dinov2_l_r${R}_f32_p1.engine \
        $G/dinov2_l_r${R}_f32_p2.engine $G/dinov2_l_r${R}_f32_p3.engine"
    csv="$E/preds_orin/dinov2_l_r${R}_f32_full.csv"
    if [ ! -s "$csv" ] || [ "$(wc -l < "$csv")" -ne $((NTEST + 1)) ]; then
        t0=$(date +%s)
        "$E/orin_infer_chain" "$csv" "$NTEST" "$E/inputs/real_dinov2_l_r${R}.fp16.bin" $CH \
            > "${csv%.csv}.log" 2>&1
        echo "  推論 $(( $(date +%s) - t0 ))s"
    fi
    d=$(tail -n +2 "$csv" | cut -d, -f2- | sort -u | wc -l)
    echo "  入力依存性: 相異なる出力=${d}/${NTEST} $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"
fi

touch "$E/logs/v2f32_r32.done"
echo "[$(date '+%m/%d %H:%M:%S')] ##### 完了 #####"
