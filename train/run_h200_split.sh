#!/usr/bin/env bash
# H200 参照ベンチ (ビルドと測定を分離する版).
#
# 較正実験の結果、4 GPU 同時に latency を測ると値が 12-33% 悪化し GPU 間のばらつきも
# 19% に拡大した (mnv4_r32: 単独 0.397 ms -> 並列 0.443/0.527/0.529 ms)。
# クロックは全基 1785 MHz を維持し電力も 100W 前後だったので、クロック低下ではなく
# CPU/PCIe の競合が原因。したがって **測定は必ず 1 GPU 単独・逐次**で行う。
#
# 一方エンジンビルドは 80-116 秒かかり全体の支配項だが、その所要時間は latency の
# 測定値に影響しない。そこで:
#   Phase 1: 4 GPU 並列でビルドのみ (--buildOnly)
#   Phase 2: GPU0 単独・逐次で測定のみ (--loadEngine)
# とすることで、正確性を保ったまま 2.6 h -> 50 分に短縮する。
#
# 起動: nohup setsid bash train/run_h200_split.sh > logs/h200_split.log 2>&1 &
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

# ---------- Phase 1: 4 GPU 並列でビルド ----------
echo "[$(date '+%H:%M:%S')] ===== Phase 1: ビルド (4 GPU 並列) ====="
build_one() {
    local gpu="$1" tag="$2"
    local onnx="$E/onnx/${tag}.onnx" eng="$E/engines_h200/${tag}.engine"
    [ -f "$eng" ] && return 0
    [ -f "$E/results_h200/${tag}.json" ] && return 0
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu $APP $TRTEXEC --onnx="$onnx" --saveEngine="$eng" --fp16 \
        --memPoolSize=workspace:4096M --buildOnly > "$E/logs_h200/build_${tag}.log" 2>&1
    local rc=$?
    echo "  [build gpu${gpu}] ${tag} rc=${rc} $(( $(date +%s)-t0 ))s"
}

i=0
for mo in $MODELS; do
    for r in $RES; do
        tag="${mo}_r${r}"
        [ -f "$E/onnx/${tag}.onnx" ] || continue
        build_one $((i % 4)) "$tag" &
        i=$((i + 1))
        # 4 本走ったら待つ (GPU あたり 1 プロセス)
        [ $((i % 4)) -eq 0 ] && wait
    done
done
wait
echo "[$(date '+%H:%M:%S')] ビルド完了: $(ls $E/engines_h200/*.engine 2>/dev/null | wc -l) 本"

# ---------- Phase 2: GPU0 単独・逐次で測定 ----------
echo "[$(date '+%H:%M:%S')] ===== Phase 2: latency 測定 (GPU0 単独・逐次) ====="
for mo in $MODELS; do
    for r in $RES; do
        tag="${mo}_r${r}"
        eng="$E/engines_h200/${tag}.engine"
        [ -f "$eng" ] || { echo "  [miss] $tag (ビルド失敗)"; continue; }
        [ -f "$E/results_h200/${tag}.json" ] && { echo "  [skip] $tag"; continue; }

        CUDA_VISIBLE_DEVICES=0 $APP $TRTEXEC --loadEngine="$eng" \
            --warmUp=2000 --iterations=300 --avgRuns=100 \
            > "$E/logs_h200/bench_${tag}.log" 2>&1
        sz=$(stat -c %s "$eng" 2>/dev/null || echo 0)
        bs=$(grep -oE "rc=[0-9]+ [0-9]+s" "$E/logs_h200/build_${tag}.log" 2>/dev/null | grep -oE "[0-9]+s" | tr -d s || echo 0)
        python3 - "$E/logs_h200/bench_${tag}.log" "$tag" "${bs:-0}" "$sz" "$E/results_h200/${tag}.json" << "PYEOF"
import json, re, sys
log, tag, build_sec, size, out = sys.argv[1:6]
txt = open(log, errors="ignore").read()
m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
              r"median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
              r"percentile\(99%\) = ([\d.]+) ms", txt)
gpu = dict(zip(["min","max","mean","median","p90","p95","p99"], [float(v) for v in m.groups()])) if m else {}
q = re.search(r"Throughput: ([\d.]+) qps", txt)
model, res = tag.rsplit("_r", 1)
rec = {"model": model, "res": int(res), "tag": tag, "device": "H200",
       "measured": "GPU0 単独・逐次 (並列測定は 12-33% 悪化するため不可)",
       "build_sec": int(build_sec or 0), "engine_MB": round(int(size)/1e6, 2),
       "gpu_compute_ms": gpu, "throughput_qps": float(q.group(1)) if q else None}
json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
print("  [ok ] %-18s engine=%.1fMB  mean=%.4f ms  %.0f qps"
      % (tag, rec["engine_MB"], gpu.get("mean", -1), rec["throughput_qps"] or -1))
PYEOF
        case "$mo" in dinov2_l|dinov3_l) rm -f "$eng" ;; esac
    done
done

echo "[$(date '+%H:%M:%S')] ===== 完了: $(ls $E/results_h200/*.json 2>/dev/null | wc -l) 構成 ====="
touch "$E/logs_h200/h200_bench.done"
