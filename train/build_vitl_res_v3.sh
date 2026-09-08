#!/usr/bin/env bash
# DINOv3-L を N=64/128/224 でも Orin にビルドして latency を測る.
#
# N=32 で確立した最終構成をそのまま適用する:
#   4 パートすべてを **パート単位で ORT 基本最適化を掛けた ONNX** からビルドする。
#   分割前の全体グラフを最適化しただけでは p2/p3 が定数エンジンになり、しかも遅い
#   (13.6 ms -> 6.6 ms とほぼ半減した)。
#
# DINOv2-L と違い FP32 を強制する段は無い (N=32 では全段 FP16 で正しく動いた)。
# ただし速度で判断せず、**入力を変えて出力が変わるか**を毎回確かめる。
#
# 1 パートあたり 10 分級のビルドになるので 3 水準で 2 時間前後かかる。
# usage: bash build_vitl_res_v3.sh <res...>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split
RESLIST="${*:-64 128 224}"
rm -f "$E/logs/build_vitl_res_v3.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_ms() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1; }
msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.1f", $1/1e6}'; }

for R in $RESLIST; do
    base="dinov3_l_r${R}_fp16_sim"
    echo "[$(date '+%H:%M:%S')] ===== DINOv3-L N=${R} ====="
    total=0
    ok=1
    for k in 0 1 2 3; do
        onnx="$S/${base}_p${k}_ort.onnx"
        eng="$E/engines_split/dinov3_l_r${R}_p${k}_ort.engine"
        log="$E/logs/build_v3_r${R}_p${k}.log"
        [ -f "$onnx" ] || { echo "  [miss] $onnx"; ok=0; break; }
        if [ -f "$eng" ]; then
            echo "  [skip] p${k} engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
        else
            t0=$(date +%s)
            "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 --workspace=1024 \
                --warmUp=2000 --iterations=200 --avgRuns=50 > "$log" 2>&1
            rc=$?
            echo "  p${k} rc=$rc $(( $(date +%s) - t0 ))s engine=$(msize "$eng") MB mean=$(mean_ms "$log") ms"
            [ $rc -ne 0 ] && { ok=0; grep -aE "\[E\]|Killed" "$log" | tail -2; break; }
        fi
        total=$(awk -v a="$total" -v b="$(mean_ms "$log")" 'BEGIN{printf "%.3f", a+b}')
    done
    [ $ok -eq 0 ] && { echo "  [NG] N=${R} は途中で失敗"; continue; }
    echo "  ★ N=${R} の 4 段合計 = ${total} ms"

    bin="$E/inputs/real_dinov3_l_r${R}.fp16.bin"
    if [ -f "$bin" ]; then
        "$E/orin_infer_chain" "$E/preds_orin/dinov3_l_r${R}_chain64.csv" 64 "$bin" \
            "$E/engines_split/dinov3_l_r${R}_p0_ort.engine" "$E/engines_split/dinov3_l_r${R}_p1_ort.engine" \
            "$E/engines_split/dinov3_l_r${R}_p2_ort.engine" "$E/engines_split/dinov3_l_r${R}_p3_ort.engine" \
            > "$E/logs/infer_v3_r${R}_chain64.log" 2>&1
        d=$(tail -n +2 "$E/preds_orin/dinov3_l_r${R}_chain64.csv" | cut -d, -f2- | sort -u | wc -l)
        echo "  入力依存性: 相異なる出力=${d}/64 $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"
    fi
done
echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
touch "$E/logs/build_vitl_res_v3.done"
