#!/usr/bin/env bash
# DINOv3-L の N=64/128/224 で、4 段チェーンのどの段が定数を返しているか特定する.
#
# N=32 ではパート単位の ORT 最適化で全段が正しく動いたが、64/128/224 では 3 水準とも
# チェーン出力が定数になった (相異なる出力 1/64)。DINOv2-L のときと同じ手順で、
# 打ち切りチェーン (p0 / p0+p1 / p0+p1+p2 / 全段) を順に実行し、
# **相異なる出力が 1 に落ちる最初の段**を壊れた段と判定する。
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
N=32   # 32 枚あれば定数かどうかは判定できる (ビルドを待たせないため軽くする)
rm -f "$E/logs/diag_v3_stages.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

for R in "$@"; do
    bin="$E/inputs/real_dinov3_l_r${R}.fp16.bin"
    [ -f "$bin" ] || { echo "[miss] $bin"; continue; }
    echo "===== DINOv3-L N=${R} ====="
    args=""
    for k in 0 1 2 3; do
        args="$args $E/engines_split/dinov3_l_r${R}_p${k}_ort.engine"
        out="$E/preds_orin/v3diag_r${R}_p0to${k}.csv"
        "$E/orin_infer_chain" "$out" "$N" "$bin" $args \
            > "$E/logs/v3diag_r${R}_p0to${k}.log" 2>&1
        rc=$?
        d=$(tail -n +2 "$out" | cut -d, -f2- | sort -u | wc -l)
        echo "  p0..p${k}: rc=$rc 相異なる出力=${d}/${N} $([ "$d" -le 1 ] && echo '<< ここで定数化')"
    done
done
touch "$E/logs/diag_v3_stages.done"
echo "===== 完了 ====="
