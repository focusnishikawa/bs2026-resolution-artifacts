#!/usr/bin/env bash
# H200 参照ベンチ v2: 4 GPU 並列 + 測定条件を強化した版.
#
# 較正で分かったこと (2026-08-17):
#   - 4 GPU 同時測定でも単独測定と有意差なし。クロックは全基 1785 MHz を維持し電力も
#     98-102 W (TDP 700 W) と余裕。当初「並列は 12-33% 悪化」と判断したのは誤りで、
#     裏でビルドが走っていた状態と単独 1 回きりの値を比べていたための誤認だった。
#   - 一方 **測定そのもののばらつきが大きい**。GPU0 単独 3 回で 0.807/0.435/0.437 ms と
#     1.9 倍も揺れた。H200 では 1 推論が 0.4 ms 程度と短く、計測ノイズが信号を上回る。
#
# したがって v2 では:
#   - 4 GPU 並列を採用 (ビルド・測定とも)
#   - warmUp 2000 -> 5000、iterations 300 -> 3000、duration 既定 3s -> 10s
#   - **各構成を 3 回測って中央値を採用**し、3 回の値も全部残す
#
# 起動: nohup setsid bash train/run_h200_v2.sh > logs/h200_v2.log 2>&1 &
set -u

P=/work/gfsi/ufsi0002/bs2026-resolution
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --bind /work --bind /home1 $SIF"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
NREP=3

mkdir -p "$E/engines_h200" "$E/logs_h200" "$E/results_h200"
cd "$P" || exit 1

# ---------- Phase 1: 4 GPU 並列でビルド ----------
echo "[$(date '+%H:%M:%S')] ===== Phase 1: ビルド (4 GPU 並列) ====="
i=0
for mo in $MODELS; do
    for r in $RES; do
        tag="${mo}_r${r}"
        [ -f "$E/onnx/${tag}.onnx" ] || continue
        [ -f "$E/engines_h200/${tag}.engine" ] && continue
        gpu=$((i % 4))
        (
            t0=$(date +%s)
            CUDA_VISIBLE_DEVICES=$gpu $APP $TRTEXEC --onnx="$E/onnx/${tag}.onnx" \
                --saveEngine="$E/engines_h200/${tag}.engine" --fp16 \
                --memPoolSize=workspace:4096M --skipInference \
                > "$E/logs_h200/build_${tag}.log" 2>&1
            echo "  [build gpu${gpu}] ${tag} rc=$? $(( $(date +%s)-t0 ))s"
        ) &
        i=$((i + 1))
        [ $((i % 4)) -eq 0 ] && wait
    done
done
wait
echo "[$(date '+%H:%M:%S')] ビルド完了: $(ls $E/engines_h200/*.engine 2>/dev/null | wc -l) 本"

# ---------- Phase 2: 4 GPU 並列で測定 (各構成 NREP 回) ----------
echo "[$(date '+%H:%M:%S')] ===== Phase 2: 測定 (4 GPU 並列 / 各 ${NREP} 回・中央値) ====="

measure_one() {
    local gpu="$1" tag="$2"
    local eng="$E/engines_h200/${tag}.engine"
    [ -f "$eng" ] || { echo "  [miss] $tag"; return; }
    [ -f "$E/results_h200/${tag}.json" ] && { echo "  [skip] $tag"; return; }
    for k in $(seq 1 $NREP); do
        CUDA_VISIBLE_DEVICES=$gpu $APP $TRTEXEC --loadEngine="$eng" \
            --warmUp=5000 --duration=10 --iterations=3000 --avgRuns=100 \
            > "$E/logs_h200/bench_${tag}_rep${k}.log" 2>&1
    done
    local sz=$(stat -c %s "$eng" 2>/dev/null || echo 0)
    python3 - "$E" "$tag" "$sz" "$NREP" << "PYEOF"
import json, re, sys, os, statistics as st
E, tag, size, nrep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
means, qps, alls = [], [], []
for k in range(1, nrep + 1):
    p = os.path.join(E, "logs_h200", "bench_%s_rep%d.log" % (tag, k))
    txt = open(p, errors="ignore").read() if os.path.exists(p) else ""
    m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
                  r"median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
                  r"percentile\(99%\) = ([\d.]+) ms", txt)
    if m:
        d = dict(zip(["min","max","mean","median","p90","p95","p99"], [float(v) for v in m.groups()]))
        means.append(d["mean"]); alls.append(d)
    q = re.search(r"Throughput: ([\d.]+) qps", txt)
    if q: qps.append(float(q.group(1)))
model, res = tag.rsplit("_r", 1)
rec = {"model": model, "res": int(res), "tag": tag, "device": "H200",
       "n_rep": len(means), "engine_MB": round(size/1e6, 2),
       "gpu_compute_mean_ms_median_of_reps": round(st.median(means), 5) if means else None,
       "gpu_compute_mean_ms_all_reps": means,
       "gpu_compute_mean_ms_spread_pct": round((max(means)-min(means))/min(means)*100, 1) if len(means) > 1 else None,
       "throughput_qps_median": round(st.median(qps), 1) if qps else None,
       "detail_last_rep": alls[-1] if alls else {},
       "measure_cond": "warmUp=5000 duration=10 iterations=3000 avgRuns=100, 4GPU 並列, %d 回中央値" % len(means)}
json.dump(rec, open(os.path.join(E, "results_h200", tag + ".json"), "w"), indent=2, ensure_ascii=False)
print("  [ok ] %-18s %.4f ms (ばらつき %.1f%%)  %.0f qps"
      % (tag, rec["gpu_compute_mean_ms_median_of_reps"] or -1,
         rec["gpu_compute_mean_ms_spread_pct"] or 0, rec["throughput_qps_median"] or -1))
PYEOF
}

i=0
for mo in $MODELS; do
    for r in $RES; do
        measure_one $((i % 4)) "${mo}_r${r}" &
        i=$((i + 1))
        [ $((i % 4)) -eq 0 ] && wait
    done
done
wait

# ViT-L のエンジンは巨大なので測り終えたら消す
rm -f "$E"/engines_h200/dinov2_l_*.engine "$E"/engines_h200/dinov3_l_*.engine

echo "[$(date '+%H:%M:%S')] ===== 完了: $(ls $E/results_h200/*.json 2>/dev/null | wc -l) 構成 ====="
touch "$E/logs_h200/h200_bench.done"
