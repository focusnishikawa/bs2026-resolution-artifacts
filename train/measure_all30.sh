#!/usr/bin/env bash
# Orin の latency を **30 回測定**へ増やし、中央値だけでなく分布そのものを記録する.
#
# 経緯: 従来は 1 構成 3 回測定の中央値だった。3 点では中央値の安定性しか言えず、
#       ばらつき (標準偏差・信頼区間) を論文の表に出せない。回数を 30 へ増やす。
#
# 測定プロトコルは 3 回版と同一で、**回数だけ**を変える (統制を崩さない):
#   CNN/ViT-S : --loadEngine --warmUp=2000 --iterations=300 --avgRuns=100
#   ViT-L     : --loadEngine --warmUp=2000 --iterations=200 --avgRuns=50
# trtexec を 30 回**別プロセスで**起動する。1 プロセス内で反復回数を増やすのとは違い、
# エンジンのロードやクロック状態を含めた実行間のばらつきを捉えられる。
#
# 出力 (既存の 3 回版 results_v2/ ・ meas_*.median は残す):
#   CNN   : results_v30/<tag>.json          30 値と統計量
#   ViT-L : logs/meas30_<tag>_r<N>_<part>.{log,stats,median}
#           .median は 3 回版と同じ 1 行形式 (中央値) にしてあるので後段がそのまま読める
#
# 再開: .stats (ViT-L) / .json (CNN) が既にあればスキップするので、同じコマンドで再開できる。
#
# usage (Orin):
#   nohup setsid bash measure_all30.sh cnn  > logs/meas30_cnn.log  2>&1 &
#   nohup setsid bash measure_all30.sh vitl > logs/meas30_vitl.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
REPS=${REPS:-30}
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
CNN_MODELS="mnv4 effb0 resnet50 vit_small"
OUT30="$E/results_v30"
mkdir -p "$OUT30" "$E/logs"

STAGE="${1:-}"
if [ "$STAGE" != "cnn" ] && [ "$STAGE" != "vitl" ]; then
    echo "usage: bash measure_all30.sh {cnn|vitl}"
    exit 1
fi

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

# 30 値から統計量を出す。標本標準偏差・95%CI (t_{0.975,29}=2.045)・変動係数まで。
stats_of() {
    printf '%s\n' "$@" | sort -g | awk '
    { v[NR] = $1; s += $1 }
    END {
        n = NR
        if (n == 0) { print "0 0 0 0 0 0 0 0"; exit }
        mean = s / n
        for (i = 1; i <= n; i++) { d = v[i] - mean; ss += d * d }
        sd = (n > 1) ? sqrt(ss / (n - 1)) : 0
        med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
        q1 = v[int(n * 0.25) + 1]; q3 = v[int(n * 0.75)]
        ci = (n > 1) ? 2.045 * sd / sqrt(n) : 0
        cv = (mean > 0) ? sd / mean * 100 : 0
        printf "%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.3f", mean, sd, med, v[1], v[n], q1, q3, cv
        printf " %.5f", ci
    }'
}

# run30 <engine> <logprefix> <trtexec 追加オプション>
# 30 回測定して "mean sd median min max q1 q3 cv ci" と全 30 値を返す
run30() {
    local eng="$1" pfx="$2" opt="$3"
    local vals=() v
    : > "${pfx}.log"
    for rep in $(seq 1 "$REPS"); do
        "$TRTEXEC" --loadEngine="$eng" $opt >> "${pfx}.log" 2>&1
        v=$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "${pfx}.log" | tail -1 | grep -oE "[0-9.]+$")
        [ -n "$v" ] && vals+=("$v")
    done
    printf '%s\n' "${vals[@]}" > "${pfx}.vals"
    stats_of "${vals[@]}"
}

# ===================== 段階 1: CNN / ViT-S (56 構成) =====================
if [ "$STAGE" = "cnn" ]; then
    OPT="--warmUp=2000 --iterations=300 --avgRuns=100"
    echo "[$(date '+%m/%d %H:%M:%S')] ===== CNN/ViT-S 56 構成 x ${REPS} 回 ====="
    for m in $CNN_MODELS; do
        for r in $RES; do
            tag="${m}_r${r}"
            json="$OUT30/${tag}.json"
            if [ -f "$json" ]; then echo "  [skip] $tag"; continue; fi
            eng="$E/engines/${tag}.engine"
            # N=64 の ViT-S は FP32 のままビルドされていたので再ビルド版が正
            if [ "$tag" = "vit_small_r64" ] && [ -f "$E/engines/vit_small_r64_rebuild.engine" ]; then
                eng="$E/engines/vit_small_r64_rebuild.engine"
            fi
            [ -f "$eng" ] || { echo "  [miss] $tag"; continue; }
            sz=$(stat -c %s "$eng" | awk '{printf "%.2f", $1/1e6}')
            read -r mean sd med mn mx q1 q3 cv ci <<EOF
$(run30 "$eng" "$E/logs/meas30_${tag}" "$OPT")
EOF
            vals=$(tr '\n' ',' < "$E/logs/meas30_${tag}.vals" | sed 's/,$//')
            cat > "$json" <<EOF
{
  "model": "$m",
  "res": $r,
  "tag": "$tag",
  "engine": "$(basename "$eng")",
  "engine_MB": $sz,
  "reps": $REPS,
  "gpu_compute_mean_ms_all_reps": [$vals],
  "mean_ms": $mean,
  "sd_ms": $sd,
  "ci95_halfwidth_ms": $ci,
  "median_ms": $med,
  "min_ms": $mn,
  "max_ms": $mx,
  "q1_ms": $q1,
  "q3_ms": $q3,
  "cv_pct": $cv,
  "protocol": "loadEngine, warmUp=2000, iterations=300, avgRuns=100, ${REPS} reps"
}
EOF
            printf "  %-16s %7s MB  mean=%s +-%s ms (CV %s%%)  median=%s\n" "$tag" "$sz" "$mean" "$sd" "$cv" "$med"
        done
    done
    touch "$E/logs/meas30_cnn.done"
    echo "[$(date '+%m/%d %H:%M:%S')] ===== CNN 完了 ====="
    exit 0
fi

# ===================== 段階 2: ViT-L (126 パート) =====================
# 配備構成のエンジン命名 (実態に合わせる):
#   DINOv2-L N=32  : engines_split/dinov2_l_r32_<part>.engine   (p3s1 は _fp32 への symlink)
#   DINOv2-L N!=32 : engines_split/dinov2_l_r<N>_raw_<part>.engine  (生の分割 = 配備構成)
#   DINOv3-L       : engines_split/dinov3_l_r<N>_<part>_fp32.engine (全段 FP32)
OPT="--warmUp=2000 --iterations=200 --avgRuns=50"
echo "[$(date '+%m/%d %H:%M:%S')] ===== ViT-L 126 パート x ${REPS} 回 ====="
for r in $RES; do
    echo "[$(date '+%m/%d %H:%M:%S')] --- N=${r} ---"
    for part in p0 p1 p2 p3s0 p3s1; do
        if [ "$r" = "32" ]; then
            tag="v2"; eng="$E/engines_split/dinov2_l_r32_${part}.engine"
        else
            tag="v2raw"; eng="$E/engines_split/dinov2_l_r${r}_raw_${part}.engine"
        fi
        pfx="$E/logs/meas30_${tag}_r${r}_${part}"
        [ -f "${pfx}.stats" ] && { echo "  [skip] ${tag} r${r} ${part}"; continue; }
        [ -f "$eng" ] || { echo "  [miss] $(basename "$eng")"; continue; }
        sz=$(stat -c %s "$eng" | awk '{printf "%.0f", $1/1e6}')
        read -r mean sd med mn mx q1 q3 cv ci <<EOF
$(run30 "$eng" "$pfx" "$OPT")
EOF
        echo "$mean $sd $med $mn $mx $q1 $q3 $cv $ci" > "${pfx}.stats"
        echo "$med" > "${pfx}.median"
        printf "  %-6s %-5s %4s MB  mean=%s +-%s (CV %s%%)  median=%s\n" "$part" "$tag" "$sz" "$mean" "$sd" "$cv" "$med"
    done
    for part in p0 p1 p2 p3; do
        eng="$E/engines_split/dinov3_l_r${r}_${part}_fp32.engine"
        pfx="$E/logs/meas30_v3_r${r}_${part}"
        [ -f "${pfx}.stats" ] && { echo "  [skip] v3 r${r} ${part}"; continue; }
        [ -f "$eng" ] || { echo "  [miss] $(basename "$eng")"; continue; }
        sz=$(stat -c %s "$eng" | awk '{printf "%.0f", $1/1e6}')
        read -r mean sd med mn mx q1 q3 cv ci <<EOF
$(run30 "$eng" "$pfx" "$OPT")
EOF
        echo "$mean $sd $med $mn $mx $q1 $q3 $cv $ci" > "${pfx}.stats"
        echo "$med" > "${pfx}.median"
        printf "  %-6s %-5s %4s MB  mean=%s +-%s (CV %s%%)  median=%s\n" "$part" "v3" "$sz" "$mean" "$sd" "$cv" "$med"
    done
done
touch "$E/logs/meas30_vitl.done"
echo "[$(date '+%m/%d %H:%M:%S')] ===== ViT-L 完了 ====="
