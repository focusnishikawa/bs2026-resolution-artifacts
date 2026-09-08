#!/usr/bin/env bash
# CNN 56 構成 (4 モデル x 14 解像度) を TensorRT 10.3 で再ビルドする。
#
# 背景: JetPack 6.2 化で TensorRT が 8.5.2 -> 10.3.0 になり、既存エンジンは
#       "magicTag == rt::kPLAN_MAGIC_TAG failed" で読めなくなった (実測済み)。
#       論文の 56 構成を測り直すには、まず同じ ONNX から再ビルドする必要がある。
#
# 起動: nohup setsid bash build_trt10_cnn56.sh > logs/build_trt10_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/build_trt10_master.log     完了マーカー: logs/build_trt10.done

set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
mkdir -p engines_trt10 logs
rm -f "$E/logs/build_trt10.done"

TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="effb0 mnv4 resnet50 vit_small"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"

total=0; built=0; skipped=0; failed=0
T_ALL0=$(date +%s)

echo "=== TRT10 再ビルド開始 $(date '+%F %T') ==="
"$TRTEXEC" --version 2>&1 | grep -i "TensorRT version" || true
echo

for m in $MODELS; do
  for r in $RES; do
    total=$((total + 1))
    onnx="onnx/${m}_r${r}.onnx"
    eng="engines_trt10/${m}_r${r}.engine"

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
    if timeout 1800 "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 \
         > "logs/build_${m}_r${r}.log" 2>&1; then
      t1=$(date +%s)
      sz=$(stat -c %s "$eng" 2>/dev/null)
      echo "[$(date '+%H:%M:%S')] OK           ${m}_r${r}  $((t1-t0)) s  ${sz} B"
      built=$((built + 1))
    else
      t1=$(date +%s)
      echo "[$(date '+%H:%M:%S')] FAILED       ${m}_r${r}  $((t1-t0)) s  -> logs/build_${m}_r${r}.log"
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
echo "エンジン数: $(ls engines_trt10/*.engine 2>/dev/null | wc -l)"

touch "$E/logs/build_trt10.done"
