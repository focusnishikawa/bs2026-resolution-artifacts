#!/usr/bin/env bash
# ViT-L の分割チェーンに **テスト集合 1,882 枚全部**を流し、精度・argmax 一致率を出せるようにする.
#
# これまで新規解像度では 64 枚の入力依存性チェックしかしていなかったので、表の精度欄が
# 空いていた。エンジンは既にあるので推論を回すだけで埋まる (1 水準あたり 1-4 分)。
#
# 出力: preds_orin/<model>_r<R>_full.csv (idx, l0..l5, argmax)
#
# usage: bash infer_vitl_all.sh <res...>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
N=1882
rm -f "$E/logs/infer_vitl_all.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

run() {
    local tag="$1" csv="$2" bin="$3"; shift 3
    for e in "$@"; do
        [ -f "$e" ] || { echo "  [skip] ${tag}: エンジンなし $(basename "$e")"; return; }
    done
    [ -f "$bin" ] || { echo "  [skip] ${tag}: 入力なし"; return; }
    if [ -s "$csv" ] && [ "$(wc -l < "$csv")" -eq $((N + 1)) ]; then
        echo "  [skip] ${tag}: 既存 ($(wc -l < "$csv") 行)"
    else
        local t0; t0=$(date +%s)
        "$E/orin_infer_chain" "$csv" "$N" "$bin" "$@" > "${csv%.csv}.log" 2>&1
        echo "  ${tag}: $(( $(date +%s) - t0 ))s $(wc -l < "$csv") 行"
    fi
    local d; d=$(tail -n +2 "$csv" | cut -d, -f2- | sort -u | wc -l)
    echo "    相異なる出力=${d}/${N} $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"
}

for R in "$@"; do
    echo "[$(date '+%H:%M:%S')] ===== N=${R} 全テスト推論 ====="
    G=$E/engines_split
    run "DINOv2-L" "$E/preds_orin/dinov2_l_r${R}_full.csv" "$E/inputs/real_dinov2_l_r${R}.fp16.bin" \
        "$G/dinov2_l_r${R}_p0.engine" "$G/dinov2_l_r${R}_p1.engine" "$G/dinov2_l_r${R}_p2.engine" \
        "$G/dinov2_l_r${R}_p3s0.engine" "$G/dinov2_l_r${R}_p3s1.engine"
    run "DINOv3-L" "$E/preds_orin/dinov3_l_r${R}_full.csv" "$E/inputs/real_dinov3_l_r${R}.fp16.bin" \
        "$G/dinov3_l_r${R}_p0_fp32.engine" "$G/dinov3_l_r${R}_p1_fp32.engine" \
        "$G/dinov3_l_r${R}_p2_fp32.engine" "$G/dinov3_l_r${R}_p3_fp32.engine"
done
touch "$E/logs/infer_vitl_all.done"
echo "[$(date '+%H:%M:%S')] ===== 推論完了 ====="
