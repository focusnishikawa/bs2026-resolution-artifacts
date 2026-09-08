#!/usr/bin/env bash
# 既にビルド済みの ViT-L 分割エンジンを --loadEngine で 3 回ずつ測り直す.
#
# 目的: 他の 56 構成と測定手順を揃える。報告書・論文の測定手順は
#       「3 回実行の中央値」なので、ViT-L だけビルド時の 1 回計測のままにしない。
#       エンジンは再利用するので測り直しは安い (1 パート数十秒)。
#
# 出力: logs/meas_v2_r<R>_<part>.log / logs/meas_v3_r<R>_p<k>.log に 3 回ぶんを追記。
#       collect_vitl_edge.py はこのログがあればそちらの中央値を使う。
#
# usage: bash measure_vitl_all.sh <res...>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
OPT="--warmUp=2000 --iterations=200 --avgRuns=50"
rm -f "$E/logs/meas_vitl_all.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

means() { grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$1" | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | tr '\n' ' '; }
median3() { printf '%s\n' "$@" | sort -g | sed -n 2p; }

# rep3 <engine> <log>
rep3() {
    local eng="$1" log="$2"
    if [ ! -f "$eng" ]; then echo "  [skip] エンジンなし $(basename "$eng")"; return; fi
    : > "$log"
    local v=()
    for rep in 1 2 3; do
        "$TRTEXEC" --loadEngine="$eng" $OPT >> "$log" 2>&1
        v+=("$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")")
    done
    local m; m=$(median3 "${v[@]}")
    local rng; rng=$(printf '%s\n' "${v[@]}" | sort -g | awk 'NR==1{a=$1}END{printf "%.1f", ($1-a)/a*100}')
    echo "  $(basename "$eng" .engine): ${v[*]} -> 中央値 ${m} ms (範囲 ${rng}%)"
    echo "$m" >> "${log%.log}.median"
}

for R in "$@"; do
    echo "[$(date '+%H:%M:%S')] ===== N=${R} 測定 ====="
    for part in p0 p1 p2 p3s0 p3s1; do
        eng="$E/engines_split/dinov2_l_r${R}_${part}.engine"
        # N=32 だけ初期の命名を使っている
        [ -f "$eng" ] || eng="$E/engines_split/dinov2_l_r${R}_${part}_fp32.engine"
        rm -f "$E/logs/meas_v2_r${R}_${part}.median"
        rep3 "$eng" "$E/logs/meas_v2_r${R}_${part}.log"
    done
    for k in 0 1 2 3; do
        rm -f "$E/logs/meas_v3_r${R}_p${k}.median"
        rep3 "$E/engines_split/dinov3_l_r${R}_p${k}_fp32.engine" "$E/logs/meas_v3_r${R}_p${k}.log"
    done
done
touch "$E/logs/meas_vitl_all.done"
echo "[$(date '+%H:%M:%S')] ===== 測定完了 ====="
