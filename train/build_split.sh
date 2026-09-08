#!/usr/bin/env bash
# 経路 A: 分割した ViT-L の各パートを Orin で順次 TRT ビルドし latency を測る.
#
# DINOv2-L は ORT 最適化を通しても MYELIN の全体融合が再発し OOM kill (rc=137, RAM ピーク
# 6363/6481MB) される。ブロック境界で 4 分割すれば融合単位が 1/4 になりピークが下がるはず。
#
# パート間で受け渡すテンソルは 1 本だけ (各 block 出口がグラフのくびれになっている)。
# 実運用では 4 エンジンを同一ストリームで連続 enqueue すればコピーは不要なので、
# ここでは各パートの GPU Compute Time を測って合計を全体の推定値とする。
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
TAG="${1:-dinov2_l_r32}"
NP="${2:-4}"
WS="${3:-1024}"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mkdir -p "$E/engines_split"
total=0
ok=0
echo "[$(date '+%H:%M:%S')] ===== $TAG を $NP 分割でビルド開始 ====="

for k in $(seq 0 $((NP-1))); do
    onnx="$E/onnx_split/${TAG}_p${k}.onnx"
    eng="$E/engines_split/${TAG}_p${k}.engine"
    log="$E/logs/split_${TAG}_p${k}.log"
    [ -f "$onnx" ] || { echo "  [miss] $onnx"; continue; }

    # 冪等にする。Orin は TRT ビルド中に高負荷で ssh が切れることがあり、
    # 途中で止まった場合は残りのパートだけ続きから流せるようにする。
    if [ -f "$eng" ] && [ -s "$eng" ]; then
        ms=$(grep -oE "GPU Compute Time: min = [0-9.]+ ms, max = [0-9.]+ ms, mean = [0-9.]+" "$log" 2>/dev/null \
             | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
        echo "  [skip] p$k 既存 engine=$(stat -c %s "$eng" | awk '{printf "%.0f MB", $1/1e6}') mean=${ms:-?} ms"
        ok=$((ok+1))
        total=$(awk -v a="$total" -v b="${ms:-0}" 'BEGIN{printf "%.4f", a+b}')
        continue
    fi

    t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 --workspace="$WS" \
        --warmUp=2000 --iterations=300 --avgRuns=100 > "$log" 2>&1
    rc=$?
    t1=$(date +%s)

    if [ $rc -ne 0 ]; then
        echo "  [NG ] p$k rc=$rc ($((t1-t0))s)  $([ $rc -eq 137 ] && echo '= OOM kill')"
        grep -iE "\[E\]|killed" "$log" | tail -2
        continue
    fi
    ok=$((ok+1))
    sz=$(stat -c %s "$eng" 2>/dev/null || echo 0)
    ms=$(grep -oE "GPU Compute Time: min = [0-9.]+ ms, max = [0-9.]+ ms, mean = [0-9.]+" "$log" \
         | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
    qps=$(grep -oE "Throughput: [0-9.]+" "$log" | grep -oE "[0-9.]+" | tail -1)
    myelin=$(grep -c "GpuLayer] MYELIN" "$log" 2>/dev/null || echo 0)
    nlayer=$(grep -c "GpuLayer]" "$log" 2>/dev/null || echo 0)
    echo "  [ok ] p$k build=$((t1-t0))s engine=$(echo $sz | awk '{printf "%.0f MB", $1/1e6}') mean=${ms} ms qps=${qps} (GpuLayer ${nlayer} / MYELIN ${myelin})"
    total=$(awk -v a="$total" -v b="${ms:-0}" 'BEGIN{printf "%.4f", a+b}')
done

echo "[$(date '+%H:%M:%S')] ===== 完了 $ok/$NP パート成功 ====="
if [ "$ok" -eq "$NP" ]; then
    echo "  ★★ 4 パート合計 GPU Compute Time = ${total} ms"
fi
free -m | awk 'NR==2{printf "  終了時メモリ: used=%s MB available=%s MB\n", $3, $7}'
