#!/usr/bin/env bash
# Orin 実機で ViT-L の TRT FP16 推論を全数 (1882 枚) 実行する.
#
# 目的は 2 つ:
#   (1) 分割した 4 エンジンの連鎖が、サーバ FP32 (results/T1_condB/preds/*.npz) と
#       どれだけ一致するか。結論 4「ViT は FP16 で壊れる」が ViT-L でも起きるかを見る
#   (2) DINOv3-L は単体エンジンもあるので、単体と 4 分割で出力が一致することを実機でも確認する
#
# Orin は TRT 実行中に ssh が切れることがあるので setsid nohup で起動し、
# 進捗は NFS 共有 (/home1) 越しに HPC から読む。
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1

echo "[$(date '+%H:%M:%S')] ===== ViT-L 実機推論 開始 ====="

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

# --- コンパイル ---
if [ ! -x "$E/orin_infer_chain" ] || [ "$E/orin_infer_chain.cpp" -nt "$E/orin_infer_chain" ]; then
    echo "[build] orin_infer_chain をコンパイル"
    # 出力を head へパイプしてはいけない。TRT 8.5 の API は deprecated 警告を大量に出すため
    # head が先に閉じて g++ が SIGPIPE で死に、コンパイル失敗と誤認する。
    g++ "$E/orin_infer_chain.cpp" -O2 -std=c++14 -Wno-deprecated-declarations \
        -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include \
        -L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64 \
        -lnvinfer -lcudart -o "$E/orin_infer_chain" > "$E/logs/compile_chain.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -x "$E/orin_infer_chain" ]; then
        echo "[NG] コンパイル失敗 rc=$rc"
        grep -E "error|undefined" "$E/logs/compile_chain.log" | head -10
        exit 1
    fi
    echo "[build] ok"
fi

mkdir -p "$E/preds_orin"
N=1882

# --- 4 分割チェーン ---
for m in dinov2_l dinov3_l; do
    bin="$E/inputs/real_${m}_r32.fp16.bin"
    out="$E/preds_orin/${m}_r32_split4_FP16.csv"
    [ -f "$bin" ] || { echo "[miss] $bin"; continue; }
    echo "[$(date '+%H:%M:%S')] $m 4 分割チェーン $N 枚"
    t0=$(date +%s)
    "$E/orin_infer_chain" "$out" "$N" "$bin" \
        "$E/engines_split/${m}_r32_p0.engine" \
        "$E/engines_split/${m}_r32_p1.engine" \
        "$E/engines_split/${m}_r32_p2.engine" \
        "$E/engines_split/${m}_r32_p3.engine" \
        > "$E/logs/infer_${m}_split4.log" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "  rc=$rc ($((t1-t0)) 秒) 行数=$(wc -l < "$out" 2>/dev/null || echo 0)"
    [ $rc -ne 0 ] && tail -5 "$E/logs/infer_${m}_split4.log"
done

# --- DINOv3-L は単体エンジンとも突き合わせる ---
single="$E/engines/dinov3_l_r32_fp16_sim.engine"
if [ -f "$single" ]; then
    echo "[$(date '+%H:%M:%S')] dinov3_l 単体エンジン $N 枚"
    t0=$(date +%s)
    "$E/orin_infer_chain" "$E/preds_orin/dinov3_l_r32_single_FP16.csv" "$N" \
        "$E/inputs/real_dinov3_l_r32.fp16.bin" "$single" \
        > "$E/logs/infer_dinov3_l_single.log" 2>&1
    echo "  rc=$? ($(( $(date +%s) - t0 )) 秒)"
fi

echo "[$(date '+%H:%M:%S')] ===== 完了 ====="
ls -la "$E/preds_orin/"*FP16.csv 2>/dev/null | awk '{printf "  %s %s\n", $5, $9}'
touch "$E/logs/vitl_infer.done"
