#!/usr/bin/env bash
# Orin Nano で ViT-L の TensorRT エンジンビルドを試す (経路 B の検証).
#
# 経路 B = ONNX から静的に決まる If を畳んだ *_sim.onnx を使う。前回 DINOv3-L は
# `IIfConditionalOutputLayer inputs must have the same shape` でパース自体に失敗していたが、
# If が消えたことでパースは通るはず。その先で DINOv2-L と同じ MYELIN の OOM に当たるかを見る。
#
# fgpu0 は重複実行禁止なので、競合がいたら何もせず終了する。
# tegrastats を併走させ、ビルド中のメモリのピークを記録する (OOM までの余裕を測るため)。
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
TAG="${1:-dinov3_l_r32_fp16_sim}"
WS="${2:-1024}"
LOG="$E/logs/buildB_${TAG}_ws${WS}.log"
TEGRA="$E/logs/buildB_${TAG}_ws${WS}.tegra"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then
    echo "[abort] trtexec が $n 本動作中 (fgpu0 は重複実行禁止)"
    exit 1
fi

onnx="$E/onnx/${TAG}.onnx"
if [ ! -f "$onnx" ]; then
    echo "[abort] $onnx が無い"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] build 開始 tag=$TAG ws=${WS}MB onnx=$(stat -c %s "$onnx" | awk '{printf "%.0f MB", $1/1e6}')"
free -m | awk 'NR==2{printf "  開始時メモリ: used=%s MB available=%s MB\n", $3, $7}'

/usr/bin/tegrastats --interval 1000 > "$TEGRA" 2>&1 &
TPID=$!

t0=$(date +%s)
"$TRTEXEC" --onnx="$onnx" --saveEngine="$E/engines/${TAG}.engine" --fp16 \
    --workspace="$WS" --buildOnly > "$LOG" 2>&1
rc=$?
t1=$(date +%s)

kill $TPID 2>/dev/null
sleep 1

echo "[$(date '+%H:%M:%S')] 終了 rc=$rc ($((t1-t0)) 秒)"
if [ $rc -eq 0 ]; then
    sz=$(stat -c %s "$E/engines/${TAG}.engine" 2>/dev/null || echo 0)
    echo "  ★ビルド成功 engine=$(echo $sz | awk '{printf "%.1f MB", $1/1e6}')"
else
    echo "  失敗 (137=OOM kill)"
    grep -iE "^\[.*\] \[E\]|error|killed" "$LOG" | tail -5
fi

# tegrastats の RAM x/y をピーク集計する
awk '{for(i=1;i<=NF;i++) if($i=="RAM"){split($(i+1),a,"/"); if(a[1]+0>m) m=a[1]+0; t=a[2]}} END{printf "  RAM ピーク: %d / %s MB\n", m, t}' "$TEGRA"
echo "  ログ: $LOG"
