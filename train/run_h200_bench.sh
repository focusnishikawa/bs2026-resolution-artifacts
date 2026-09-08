#!/usr/bin/env bash
# Phase 3 補完: H200 上で同じ ONNX から TensorRT FP16 エンジンを作り latency を測る.
#
# 目的は 2 つ:
#   1. エッジ (Orin Nano) とサーバ (H200) の比を示す
#   2. **Orin では載らなかった ViT-L 2 種の速度を押さえる** (DINOv2-L は OOM、DINOv3-L は If 非対応)
#
# trtexec は SIF 内の /usr/src/tensorrt/bin/trtexec を使う。
# H200 は 143 GB あるので FP32 ONNX のままで問題ない。
#
# 起動: nohup setsid bash train/run_h200_bench.sh > logs/h200_bench_master.log 2>&1 &
set -u

P=/work/gfsi/ufsi0002/bs2026-resolution
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --bind /work --bind /home1 $SIF"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"

mkdir -p "$E/engines_h200" "$E/logs_h200" "$E/results_h200"
cd "$P" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== H200 ベンチ開始 ====="
$APP $TRTEXEC --help 2>&1 | head -1

for mo in $MODELS; do
    for r in $RES; do
        tag="${mo}_r${r}"
        onnx="$E/onnx/${tag}.onnx"
        eng="$E/engines_h200/${tag}.engine"
        log="$E/logs_h200/bench_${tag}.log"
        [ -f "$onnx" ] || { echo "  [miss] $tag"; continue; }
        [ -f "$E/results_h200/${tag}.json" ] && { echo "  [skip] $tag"; continue; }

        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 $APP $TRTEXEC --onnx="$onnx" --saveEngine="$eng" --fp16 \
            --memPoolSize=workspace:4096M --warmUp=2000 --iterations=300 --avgRuns=100 \
            > "$log" 2>&1
        rc=$?
        t1=$(date +%s)
        if [ $rc -ne 0 ]; then
            echo "  [NG ] $tag rc=$rc ($((t1-t0))s)  $(grep -oE 'Parsing model failed|out of memory' "$log" | head -1)"
            rm -f "$eng"
            continue
        fi
        sz=$(stat -c %s "$eng" 2>/dev/null || echo 0)
        python3 - "$log" "$tag" "$((t1-t0))" "$sz" "$E/results_h200/${tag}.json" << "PYEOF"
import json, re, sys
log, tag, build_sec, size, out = sys.argv[1:6]
txt = open(log, errors="ignore").read()
m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
              r"median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
              r"percentile\(99%\) = ([\d.]+) ms", txt)
gpu = {}
if m:
    gpu = dict(zip(["min", "max", "mean", "median", "p90", "p95", "p99"], [float(v) for v in m.groups()]))
q = re.search(r"Throughput: ([\d.]+) qps", txt)
model, res = tag.rsplit("_r", 1)
rec = {"model": model, "res": int(res), "tag": tag, "device": "H200",
       "build_sec": int(build_sec), "engine_MB": round(int(size) / 1e6, 2),
       "gpu_compute_ms": gpu, "throughput_qps": float(q.group(1)) if q else None}
json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
print("  [ok ] %-18s build=%ss engine=%.1fMB  mean=%.3f ms  %.0f qps"
      % (tag, build_sec, rec["engine_MB"], gpu.get("mean", -1), rec["throughput_qps"] or -1))
PYEOF
        # 巨大エンジンはディスクを食うので ViT-L は測り終えたら消す
        case "$mo" in dinov2_l|dinov3_l) rm -f "$eng" ;; esac
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 完了: $(ls $E/results_h200/*.json 2>/dev/null | wc -l) 構成 ====="
touch "$E/logs_h200/h200_bench.done"
