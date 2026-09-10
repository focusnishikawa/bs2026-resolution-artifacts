#!/usr/bin/env bash
# ViT-S/16 の 14 解像度を TensorRT 10.3 で **FP32** エンジンとして構築する。
#
# 背景: これまでの Orin 実測はすべて FP16 (--fp16) である。ViT-S/16 は FP16 で
#       精度が崩れる構成があると分かっているため、FP32 でのレイテンシと配備精度を
#       同じ作法で測り直して比較する。
#
# ⭐ FP32 ビルドとは **--fp16 を付けない**こと。それ以外は build_trt10_cnn56.sh と同じ。
#    ONNX は seed42 由来の onnx/ で、FP16 レイテンシエンジンと出所が同一である
#    (エンジンの精度だけが違う状態にして、精度モードの差だけを切り出す)。
#
# 起動: nohup setsid bash build_trt10_fp32_vits.sh > logs/build_trt10_fp32_vits_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/build_trt10_fp32_vits_master.log   完了マーカー: logs/build_trt10_fp32_vits.done

set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
mkdir -p engines_trt10_fp32 logs
rm -f "$E/logs/build_trt10_fp32_vits.done"

TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="vit_small"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"

# ---- 競合チェック (他の測定・ビルドが走っていたら中止) ----
n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then
    echo "[abort] trtexec が ${n} 本動作中。ビルドか測定が走っている"
    exit 1
fi

total=0; built=0; skipped=0; failed=0
T_ALL0=$(date +%s)

echo "=== TRT10 FP32 ビルド開始 (ViT-S/16) $(date '+%F %T') ==="
"$TRTEXEC" --version 2>&1 | grep -i "TensorRT version" || true
echo

for m in $MODELS; do
  for r in $RES; do
    total=$((total + 1))
    onnx="onnx/${m}_r${r}.onnx"
    eng="engines_trt10_fp32/${m}_r${r}.engine"

    if [ ! -f "$onnx" ]; then
      echo "[$(date '+%H:%M:%S')] MISSING onnx  ${m}_r${r}"
      failed=$((failed + 1))
      continue
    fi
    if [ -s "$eng" ]; then
      echo "[$(date '+%H:%M:%S')] skip         ${m}_r${r} (既存)"
      skipped=$((skipped + 1))
      continue
    fi

    t0=$(date +%s)
    # ⭐ ここに --fp16 を付けない。これが FP32 ビルドである。
    if timeout 1800 "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" \
         > "logs/build_fp32_vits_${m}_r${r}.log" 2>&1; then
      t1=$(date +%s)
      sz=$(stat -c %s "$eng" 2>/dev/null)
      echo "[$(date '+%H:%M:%S')] OK           ${m}_r${r}  $((t1-t0)) s  ${sz} B"
      built=$((built + 1))
    else
      t1=$(date +%s)
      echo "[$(date '+%H:%M:%S')] FAILED       ${m}_r${r}  $((t1-t0)) s  -> logs/build_fp32_vits_${m}_r${r}.log"
      rm -f "$eng"
      failed=$((failed + 1))
    fi
  done
done

T_ALL1=$(date +%s)
echo
echo "=== 完了 $(date '+%F %T') ==="
echo "総数 ${total} / 新規 ${built} / スキップ ${skipped} / 失敗 ${failed}"
echo "所要 $(( (T_ALL1-T_ALL0) / 60 )) 分"
echo "エンジン数: $(ls engines_trt10_fp32/*.engine 2>/dev/null | wc -l)"

touch "$E/logs/build_trt10_fp32_vits.done"
