#!/usr/bin/env bash
# p3 エンジンが入力を無視する問題の原因を実機で切り分ける.
#
# 確定済みの事実:
#   - p3 は 1 段だけで動かしても 64 枚すべて同一の logits を返す。入力を 0.5 倍しても
#     1 ビットも変わらない = 入力を完全に無視している
#   - 同じ ONNX を ORT に通すと入力依存に正しく動く = ONNX ではなく TRT ビルドの問題
#   - 入出力の dtype/バイト数/フォーマットは正常 (fp16 / 10,240 B / kLINEAR)
#
# 唯一 p1/p2 と違うのは「p3 のグラフ全体が単一の MYELIN ForeignNode になる」こと。
# 分割で残った symbolic な batch 次元 (unk__3) が trtexec の自動オーバーライド
# (Automatically overriding shape to: 1x5x1024) を経由することが引き金と疑われるため、
# 次の 3 通りでビルドして入力依存性を確かめる:
#
#   v_static : batch 次元を静的な 1 に潰した ONNX (static_dims.py の出力)
#   v_shapes : 元の ONNX + --shapes で明示 (自動オーバーライドを介さない)
#   v_fp32   : 元の ONNX を FP32 でビルド (MYELIN の fp16 経路が原因かの切り分け)
#
# 起動 (Orin):
#   setsid nohup bash diag_p3.sh > logs/diag_p3_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
TAG=dinov2_l_r32
IN_NAME="/blocks/blocks.17/Add_1_output_0"
cd "$E" || exit 1
rm -f "$E/logs/diag_p3.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

build() {  # build <名前> <onnx> <追加オプション...>
    local name="$1" onnx="$2"; shift 2
    local eng="$E/engines_split/${TAG}_p3_${name}.engine"
    local log="$E/logs/build_p3_${name}.log"
    if [ -f "$eng" ] && [ -s "$eng" ]; then echo "[skip] $name (既存)"; return 0; fi
    echo "[$(date '+%H:%M:%S')] build $name"
    local t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --workspace=1024 \
        --warmUp=1000 --iterations=100 --avgRuns=50 "$@" > "$log" 2>&1
    local rc=$?
    echo "  rc=$rc ($(( $(date +%s) - t0 ))s) $(grep -c 'GpuLayer]' "$log" 2>/dev/null || echo 0) layers / MYELIN $(grep -c 'GpuLayer] MYELIN' "$log" 2>/dev/null || echo 0)"
    grep -a "Automatically overriding" "$log" | head -1
    return $rc
}

check() {  # check <名前>  -- mid と half で出力が変わるか
    local name="$1"
    local eng="$E/engines_split/${TAG}_p3_${name}.engine"
    [ -f "$eng" ] || { echo "  [miss] engine なし"; return 1; }
    for v in mid half; do
        "$E/orin_infer_chain" "$E/preds_orin/p3only_${name}_${v}.csv" 64 \
            "$E/inputs/${TAG}_p012_${v}.fp16.bin" "$eng" \
            > "$E/logs/infer_p3_${name}_${v}.log" 2>&1
        local f="$E/preds_orin/p3only_${name}_${v}.csv"
        echo "  $v: 相異なる出力=$(tail -n +2 "$f" | cut -d, -f2- | sort -u | wc -l) / 64  先頭=$(sed -n 2p "$f" | cut -d, -f2-4)"
    done
}

echo "===== p3 診断ビルド 開始 ====="

build static "$E/onnx_split/${TAG}_p3_static.onnx" --fp16      && check static
build shapes "$E/onnx_split/${TAG}_p3.onnx" --fp16 --shapes="${IN_NAME}":1x5x1024 && check shapes
build fp32   "$E/onnx_split/${TAG}_p3_static.onnx"             && check fp32

echo "===== 完了 ====="
touch "$E/logs/diag_p3.done"
