#!/usr/bin/env bash
# DINOv3-L の 4 段のうち **どの段が FP16 で壊れるか**を段ごとに特定する.
#
# 背景: DINOv3-L は現状「全段 FP32」でしか動かしていない。根拠は段境界の活性値が 155,118 で
#   fp16 上限 65,504 を超えることだが、これは **境界しか見ていない**。DINOv2-L では
#   「境界は 9.6 に収まるのに p3s1 の内部が 356,409」という前例があり、逆に
#   「境界が超えていても、その段の計算自体は FP16 で成立する」可能性は潰せていない。
#
# 全段 FP32 の強制は段あたり 1.6-1.8 倍のコストになる (DINOv2-L で実測) ので、
# 1 段でも FP16 にできれば DINOv3-L は速くなる。
#
# 方法: 段 k だけを FP16、残りを FP32 にしたチェーンで全数 1,882 枚を推論し、
#   **入力依存性** (相異なる出力の数) で正常性を判定する。
#   ⚠️ 速度もサイズも正常性の証拠にならない (壊れたエンジンは計算が消えるぶん小さく速い)。
#
# 対象は minif 版 (If だけ畳んだ経路)。ORT 全体最適化版より 5% 速く、グラフがほぼ無傷なため。
#
# 起動 (Orin): nohup setsid bash probe_v3_stage_fp16.sh > logs/v3_stage_fp16_master.log 2>&1 < /dev/null &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split_minif
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
MEASOPT="--warmUp=2000 --iterations=200 --avgRuns=50"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
R=32
NTEST=1882
rm -f "$E/logs/v3_stage_fp16.done"

n=$(pgrep -cx trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }
median3() { printf '%s\n' "$@" | sort -g | sed -n 2p; }

echo "[$(date '+%m/%d %H:%M:%S')] ##### DINOv3-L: 段ごとの FP16 可否を判定 (N=${R}) #####"

# ---- 1. 各段の FP16 エンジンをビルド (FP32 強制を外すだけ) ----
for k in 0 1 2 3; do
    onnx="$S/dinov3_l_r${R}_fp16_minif_p${k}.onnx"
    eng="$E/engines_split/dinov3_l_r${R}_minif_f16_p${k}.engine"
    log="$E/logs/build_v3minif_f16_r${R}_p${k}.log"
    if [ -f "$eng" ]; then echo "  [skip] p${k} FP16 $(msize "$eng") MB"; continue; fi
    [ -f "$onnx" ] || { echo "  [NG] ONNX なし $(basename "$onnx")"; continue; }
    t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT > "$log" 2>&1
    rc=$?
    echo "  p${k} FP16 rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB"
done

# ---- 2. 段 k だけ FP16 にしたチェーンで全数推論し、入力依存性で判定 ----
G=$E/engines_split
BIN="$E/inputs/real_dinov3_l_r${R}.fp16.bin"
echo ""
echo "[$(date '+%H:%M:%S')] ### 段ごとの判定 (全数 ${NTEST} 枚) ###"
printf "%-14s %10s %12s %10s\n" "構成" "合計[ms]" "相異なる出力" "判定"

for k in 0 1 2 3; do
    CH=""
    for j in 0 1 2 3; do
        if [ "$j" = "$k" ]; then CH="$CH $G/dinov3_l_r${R}_minif_f16_p${j}.engine"
        else                     CH="$CH $G/dinov3_l_r${R}_minif_p${j}.engine"; fi
    done
    csv="$E/preds_orin/dinov3_l_r${R}_minif_f16p${k}.csv"
    [ -s "$csv" ] || "$E/orin_infer_chain" "$csv" "$NTEST" "$BIN" $CH > "${csv%.csv}.log" 2>&1
    d=$(tail -n +2 "$csv" 2>/dev/null | cut -d, -f2- | sort -u | wc -l)
    tot=0
    for j in 0 1 2 3; do
        if [ "$j" = "$k" ]; then e="$G/dinov3_l_r${R}_minif_f16_p${j}.engine"; p="$E/logs/meas_v3minif_f16_r${R}_p${j}"
        else                     e="$G/dinov3_l_r${R}_minif_p${j}.engine";     p="$E/logs/meas_v3minif_r${R}_p${j}"; fi
        if [ ! -f "${p}.median" ]; then
            : > "${p}.log"; v=()
            for rep in 1 2 3; do
                "$TRTEXEC" --loadEngine="$e" $MEASOPT >> "${p}.log" 2>&1
                v+=("$(grep -aoE 'GPU Compute Time:.*mean = [0-9.]+' "${p}.log" | tail -1 | grep -oE '[0-9.]+$')")
            done
            median3 "${v[@]}" > "${p}.median"
        fi
        tot=$(awk -v a="$tot" -v b="$(cat "${p}.median")" 'BEGIN{printf "%.3f", a+b}')
    done
    verdict=$([ "$d" -le 1 ] && echo "定数=壊れた" || echo "正常")
    printf "p%s だけFP16   %10s %12s %10s\n" "$k" "$tot" "${d}/${NTEST}" "$verdict"
done

# ---- 3. 参考: 全段 FP16 (既知の壊れる構成) ----
CH=""
for j in 0 1 2 3; do CH="$CH $G/dinov3_l_r${R}_minif_f16_p${j}.engine"; done
csv="$E/preds_orin/dinov3_l_r${R}_minif_f16all.csv"
[ -s "$csv" ] || "$E/orin_infer_chain" "$csv" "$NTEST" "$BIN" $CH > "${csv%.csv}.log" 2>&1
d=$(tail -n +2 "$csv" 2>/dev/null | cut -d, -f2- | sort -u | wc -l)
printf "全段FP16       %10s %12s %10s\n" "-" "${d}/${NTEST}" "$([ "$d" -le 1 ] && echo '定数=壊れた' || echo '正常')"

touch "$E/logs/v3_stage_fp16.done"
echo "[$(date '+%m/%d %H:%M:%S')] ##### 完了 #####"
