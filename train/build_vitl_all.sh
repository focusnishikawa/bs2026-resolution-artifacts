#!/usr/bin/env bash
# ViT-L の分割チェーンを残り 10 水準ぶんまとめて Orin にビルドする.
#
# 目的: ViT-L のエッジ実測を代表 4 水準 (32/64/128/224) から **全 14 水準**へ広げる。
#
# 配備構成 (N=32 で確立し 64/128/224 でも成立を確認したもの) をそのまま適用する:
#   DINOv2-L: p0/p1/p2/p3s0 は FP16、p3s1 (最終ブロック群 + norm + head) のみ FP32 強制
#             -> 最終ブロック群の内部活性が 3.6e5 に達し fp16 上限 65,504 を超えるため
#   DINOv3-L: 4 段すべて FP32 (段境界の残差ストリームが 1.55e5 で全段が超過)
#
# 速度が出ても採用しない。**入力を変えて出力が変わるか**を各水準で必ず確かめる
# (定数エンジンは計算が消えるぶん速く出る)。
#
# 測定はこのスクリプトでは行わない (軽い設定でビルドのみ)。
# latency は build 後に measure_vitl_all.sh で --loadEngine の 3 回中央値を取る。
#
# usage: bash build_vitl_all.sh <res...>
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=$E/onnx_split
SP3=$E/onnx_split_p3
BUILDOPT="--workspace=1024 --warmUp=200 --iterations=30 --avgRuns=10"
FP32="--precisionConstraints=obey --layerPrecisions=*:fp32"
rm -f "$E/logs/build_vitl_all.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

msize() { stat -c %s "$1" 2>/dev/null | awk '{printf "%.0f", $1/1e6}'; }

# build <onnx> <engine> <log> <extra opts...>
build() {
    local onnx="$1" eng="$2" log="$3"; shift 3
    if [ -f "$eng" ]; then echo "  [skip] $(basename "$eng") $(msize "$eng") MB"; return 0; fi
    if [ ! -f "$onnx" ]; then echo "  [NG] ONNX なし $(basename "$onnx")"; return 1; fi
    local t0; t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 $BUILDOPT "$@" > "$log" 2>&1
    local rc=$?
    echo "  $(basename "$eng") rc=$rc $(( $(date +%s) - t0 ))s $(msize "$eng") MB"
    [ $rc -ne 0 ] && { grep -aE "\[E\]|Killed" "$log" | tail -2; return 1; }
    return 0
}

# 入力依存性: 実画像 64 枚を流して相異なる出力が何通りあるか
depcheck() {
    local csv="$1" bin="$2"; shift 2
    if [ ! -f "$bin" ]; then echo "  [warn] 入力 bin なし"; return; fi
    "$E/orin_infer_chain" "$csv" 64 "$bin" "$@" > "${csv%.csv}.log" 2>&1
    local d; d=$(tail -n +2 "$csv" | cut -d, -f2- | sort -u | wc -l)
    echo "  入力依存性: 相異なる出力=${d}/64 $([ "$d" -le 1 ] && echo '<< 定数 (無効)')"
}

for R in "$@"; do
    # ---- DINOv2-L: 5 段 (p3s1 のみ FP32) ----
    base="dinov2_l_r${R}_fp16_sim"
    echo "[$(date '+%H:%M:%S')] ===== DINOv2-L N=${R} ====="
    ok=1
    for part in p0 p1 p2; do
        build "$S/${base}_${part}.onnx" "$E/engines_split/dinov2_l_r${R}_${part}.engine" \
              "$E/logs/build_v2_r${R}_${part}.log" || { ok=0; break; }
    done
    if [ $ok -eq 1 ]; then
        build "$SP3/${base}_p3_p0.onnx" "$E/engines_split/dinov2_l_r${R}_p3s0.engine" \
              "$E/logs/build_v2_r${R}_p3s0.log" || ok=0
    fi
    if [ $ok -eq 1 ]; then
        build "$SP3/${base}_p3_p1.onnx" "$E/engines_split/dinov2_l_r${R}_p3s1.engine" \
              "$E/logs/build_v2_r${R}_p3s1.log" $FP32 || ok=0
    fi
    if [ $ok -eq 1 ]; then
        depcheck "$E/preds_orin/dinov2_l_r${R}_chain64.csv" "$E/inputs/real_dinov2_l_r${R}.fp16.bin" \
            "$E/engines_split/dinov2_l_r${R}_p0.engine" "$E/engines_split/dinov2_l_r${R}_p1.engine" \
            "$E/engines_split/dinov2_l_r${R}_p2.engine" "$E/engines_split/dinov2_l_r${R}_p3s0.engine" \
            "$E/engines_split/dinov2_l_r${R}_p3s1.engine"
    else
        echo "  [NG] DINOv2-L N=${R} は途中で失敗"
    fi

    # ---- DINOv3-L: 4 段すべて FP32 ----
    base="dinov3_l_r${R}_fp16_sim"
    echo "[$(date '+%H:%M:%S')] ===== DINOv3-L N=${R} ====="
    ok=1
    for k in 0 1 2 3; do
        build "$S/${base}_p${k}_ort.onnx" "$E/engines_split/dinov3_l_r${R}_p${k}_fp32.engine" \
              "$E/logs/build_v3_r${R}_p${k}_fp32.log" $FP32 || { ok=0; break; }
    done
    if [ $ok -eq 1 ]; then
        depcheck "$E/preds_orin/dinov3_l_r${R}_fp32all64.csv" "$E/inputs/real_dinov3_l_r${R}.fp16.bin" \
            "$E/engines_split/dinov3_l_r${R}_p0_fp32.engine" "$E/engines_split/dinov3_l_r${R}_p1_fp32.engine" \
            "$E/engines_split/dinov3_l_r${R}_p2_fp32.engine" "$E/engines_split/dinov3_l_r${R}_p3_fp32.engine"
    else
        echo "  [NG] DINOv3-L N=${R} は途中で失敗"
    fi
done
touch "$E/logs/build_vitl_all.done"
echo "[$(date '+%H:%M:%S')] ===== ビルド完了 ====="
