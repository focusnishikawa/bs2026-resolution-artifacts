#!/usr/bin/env bash
# Phase 3: Orin Nano 実機で TensorRT FP16 エンジンをビルドし latency を計測する.
#
# fgpu0 は重複実行禁止なので 1 構成ずつ直列に回す。
# ビルドと計測を 1 回の trtexec で行い (--buildOnly を付けない)、ログから
# ビルド時間・エンジンサイズ・GPU Compute Time (mean/median/p90/p95/p99) を抽出する。
#
# 対象は CNN 4 種 + ViT-S の 5 モデル x 14 解像度 = 70 構成。
# ViT-L 2 種は Orin では実行不可 (下記) のため除外し、H200 参照値で補う:
#   - DINOv2-L: FP16 ONNX 606 MB でも trtexec が OOM kill (rc=137)。
#     workspace 2048/1024/512/256 MB、--tacticSources 制限のいずれでも回避できず。
#     実効メモリ 6.4 GB では 300M パラメータの TRT ビルドが成立しない
#   - DINOv3-L: TensorRT 8.5.2 が RoPE 由来の `If` オペレータを解釈できずパース失敗。
#     仮に解消しても DINOv2-L と同サイズなので OOM に達する
#
# 起動: nohup setsid bash run_orin_bench.sh > logs/orin_bench_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="mnv4 effb0 resnet50 vit_small"
MODELS_ALL="mnv4 effb0 resnet50 vit_small"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"

mkdir -p "$E/engines" "$E/logs" "$E/results"
cd "$E" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== Orin ベンチ開始 ====="
echo "power mode: $(nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1)"

for mo in $MODELS_ALL; do
    for r in $RES; do
        tag="${mo}_r${r}"
        onnx="$E/onnx/${tag}.onnx"
        eng="$E/engines/${tag}.engine"
        log="$E/logs/bench_${tag}.log"
        [ -f "$onnx" ] || { echo "  [miss] $onnx"; continue; }
        if [ -f "$E/results/${tag}.json" ]; then echo "  [skip] $tag"; continue; fi

        # 競合がいないことを確認 (fgpu0 は重複実行禁止)
        n=$(pgrep -c trtexec 2>/dev/null || echo 0)
        if [ "$n" -gt 0 ]; then echo "  [wait] trtexec が $n 本動作中"; sleep 30; fi

        t0=$(date +%s)
        "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 \
            --workspace=1024 --warmUp=2000 --iterations=300 --avgRuns=100 \
            > "$log" 2>&1
        rc=$?
        t1=$(date +%s)

        if [ $rc -ne 0 ]; then
            echo "  [NG ] $tag rc=$rc ($((t1-t0))s)"
            continue
        fi

        sz=$(stat -c %s "$eng" 2>/dev/null || echo 0)
        # trtexec の出力から GPU Compute Time の統計を取る
        python3 - "$log" "$tag" "$((t1-t0))" "$sz" "$E/results/${tag}.json" << "PYEOF"
import json, re, sys
log, tag, build_sec, size, out = sys.argv[1:6]
txt = open(log, errors="ignore").read()
def grab(pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None
gpu = {}
m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
              r"median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
              r"percentile\(99%\) = ([\d.]+) ms", txt)
if m:
    keys = ["min", "max", "mean", "median", "p90", "p95", "p99"]
    gpu = {k: float(v) for k, v in zip(keys, m.groups())}
else:
    m2 = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms", txt)
    if m2:
        gpu = dict(zip(["min", "max", "mean", "median"], [float(v) for v in m2.groups()]))
model, res = tag.rsplit("_r", 1)
rec = {"model": model, "res": int(res), "tag": tag,
       "build_sec": int(build_sec), "engine_bytes": int(size),
       "engine_MB": round(int(size) / 1e6, 2),
       "gpu_compute_ms": gpu,
       "throughput_qps": grab(r"Throughput: ([\d.]+) qps"),
       "latency_mean_ms": gpu.get("mean"),
       "host_latency_mean_ms": grab(r"Latency: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms")}
json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
print("  [ok ] %-18s build=%ss engine=%.1fMB  mean=%.3f ms  %.0f qps"
      % (tag, build_sec, rec["engine_MB"], gpu.get("mean", -1), rec["throughput_qps"] or -1))
PYEOF
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 完了: $(ls $E/results/*.json 2>/dev/null | wc -l) 構成 ====="
touch "$E/logs/orin_bench.done"
