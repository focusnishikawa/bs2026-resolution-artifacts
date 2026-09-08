#!/usr/bin/env bash
# Orin の latency を全 56 構成について 3 回測り直し、H200 と同じ「3 回測定」プロトコルに揃える.
#
# 経緯: ViT-S/16 の latency が解像度に対して単調でなく、特に N=64 が前後から +39% 外れていた。
#       原因は計測ノイズではなく **N=64 のエンジンだけ FP16 経路が選ばれず FP32 のまま
#       ビルドされていた**こと (engine 86.3 MB / 他水準は 44 MB)。再ビルドで 43.8 MB・
#       2.81 ms となり前後 (2.69 / 2.83) と整合した。
#
# TRT はビルド時にカーネルを実測で選ぶため、計測が揺れるとタクティク選択も揺れる。
# 当初の Orin 実測は 1 構成 1 回だったので、H200 と同じく 3 回測定して中央値を採る形に統一し、
# 併せて engine サイズを記録して同種の取りこぼし (FP32 のまま) を検出できるようにする。
#
# 計測のみ (--loadEngine) なのでビルドはやり直さない。ただし vit_small_r64 だけは
# 再ビルドした engine を使う。
#
# 起動 (Orin):
#   setsid nohup bash remeasure_orin_all.sh > logs/remeasure_all_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="mnv4 effb0 resnet50 vit_small"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
OUT="$E/results_v2"
mkdir -p "$OUT" "$E/logs_remeasure"
rm -f "$E/logs/remeasure_all.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

echo "[$(date '+%H:%M:%S')] ===== Orin 全 56 構成を 3 回測定 ====="
for m in $MODELS; do
    for r in $RES; do
        tag="${m}_r${r}"
        eng="$E/engines/${tag}.engine"
        # N=64 の ViT-S は FP32 のままビルドされていたので再ビルド版を使う
        if [ "$tag" = "vit_small_r64" ] && [ -f "$E/engines/vit_small_r64_rebuild.engine" ]; then
            eng="$E/engines/vit_small_r64_rebuild.engine"
        fi
        [ -f "$eng" ] || { echo "  [miss] $tag"; continue; }
        sz=$(stat -c %s "$eng" | awk '{printf "%.2f", $1/1e6}')
        means=""
        for rep in 1 2 3; do
            log="$E/logs_remeasure/${tag}_v2_rep${rep}.log"
            "$TRTEXEC" --loadEngine="$eng" --warmUp=2000 --iterations=300 --avgRuns=100 \
                > "$log" 2>&1
            v=$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$log" | grep -oE "mean = [0-9.]+" \
                 | grep -oE "[0-9.]+" | tail -1)
            means="$means $v"
        done
        # 3 回の mean から中央値と広がりを出す
        read -r med spread <<EOF
$(echo $means | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END{printf "%.5f %.2f", a[2], (a[3]-a[1])/a[1]*100}')
EOF
        cat > "$OUT/${tag}.json" <<EOF
{
  "model": "$m",
  "res": $r,
  "tag": "$tag",
  "engine": "$(basename "$eng")",
  "engine_MB": $sz,
  "gpu_compute_mean_ms_all_reps": [$(echo $means | tr ' ' ',')],
  "gpu_compute_mean_ms_median_of_reps": $med,
  "gpu_compute_mean_ms_spread_pct": $spread,
  "protocol": "loadEngine, warmUp=2000, iterations=300, avgRuns=100, 3 reps"
}
EOF
        printf "  %-16s %6s MB  median=%s ms  広がり=%s%%\n" "$tag" "$sz" "$med" "$spread"
    done
done

echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
touch "$E/logs/remeasure_all.done"
