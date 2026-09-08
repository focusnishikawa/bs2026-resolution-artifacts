#!/usr/bin/env bash
# CNN 56 構成を TensorRT 10.3 エンジンで測定する。電力モードごとに実行する。
#
# ⭐ 設計方針 (ユーザー指示 2026-08-31): 論文の既存値 (JP5.1.2 / TRT 8.5.2 / 15W) は参照しない。
#    すべて新環境 (JP6.2 R36.4.3 / TRT 10.3) で測り、15W と MAXN_SUPER を比較する。
#    これで TensorRT 版が交絡せず、電力モードの差だけを切り出せる。
#
# ⭐ 測定プロトコルは既存 measure_all30.sh を踏襲:
#    trtexec --loadEngine --warmUp=2000 --iterations=300 --avgRuns=100 を
#    **30 回別プロセスで**起動し、各回の GPU Compute Time mean を集める。
#    (1 プロセス内で反復を増やすのとは違い、プロセス起動ごとのばらつきを含められる)
#
# ⭐ 本版の追加点: INA3221 が読めるようになったので **消費電力を同時記録**する
#    (VDD_IN / VDD_CPU_GPU_CV / VDD_SOC)。
#
# 使い方 (bash で起動すること):
#   bash measure_trt10_mode.sh maxn     # MAXN_SUPER (ID=2)
#   bash measure_trt10_mode.sh 15w      # 15W        (ID=0)
#
# 起動例:
#   nohup setsid bash measure_trt10_mode.sh maxn > logs/meas_trt10_maxn.log 2>&1 < /dev/null &

set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1

MODE="${1:-}"
case "$MODE" in
    maxn) MODE_ID=2; MODE_NAME="MAXN_SUPER" ;;
    15w)  MODE_ID=0; MODE_NAME="15W" ;;
    *) echo "使い方: bash $0 {maxn|15w}"; exit 1 ;;
esac

OUT="results_trt10_${MODE}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="effb0 mnv4 resnet50 vit_small"
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
REPS=30
OPT="--warmUp=2000 --iterations=300 --avgRuns=100"

mkdir -p "$OUT" logs
rm -f "$E/logs/meas_trt10_${MODE}.done"

# ---- 競合チェック (他の測定・ビルドが走っていたら中止) ----
n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then
    echo "[abort] trtexec が ${n} 本動作中。ビルドか別の測定が走っている"
    exit 1
fi

# ---- 電力モードを設定 ----
echo "=== 電力モードを ${MODE_NAME} (ID=${MODE_ID}) に設定 ==="
sudo -n nvpmodel -m "$MODE_ID" 2>&1 | head -3
sleep 15                      # クロック・コア構成が落ち着くのを待つ
echo "--- 設定後の状態 ---"
nvpmodel -q 2>&1 | head -2
echo "Online CPUs: $(nproc) / MaxFreq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)"
echo

T_ALL0=$(date +%s)
done_n=0; skip_n=0; fail_n=0

for m in $MODELS; do
  for r in $RES; do
    tag="${m}_r${r}"
    eng="engines_trt10/${tag}.engine"
    json="${OUT}/${tag}.json"

    if [ ! -s "$eng" ]; then
        echo "[$(date '+%H:%M:%S')] MISSING engine ${tag}"; fail_n=$((fail_n+1)); continue
    fi
    if [ -s "$json" ]; then
        echo "[$(date '+%H:%M:%S')] skip          ${tag} (既存)"; skip_n=$((skip_n+1)); continue
    fi

    t0=$(date +%s)
    pwrlog="/tmp/pwr_${MODE}_${tag}.log"

    # 電力記録を開始 (★これは wait の対象にしない)
    ( tegrastats --interval 500 > "$pwrlog" 2>&1 ) &
    PWRPID=$!

    vals=""
    for rep in $(seq 1 "$REPS"); do
        replog="/tmp/rep_${MODE}_${tag}.log"
        if timeout 600 "$TRTEXEC" --loadEngine="$eng" $OPT > "$replog" 2>&1; then
            v=$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$replog" | tail -1 | grep -oE "[0-9.]+$")
            [ -n "$v" ] && vals="${vals} ${v}"
        fi
    done

    kill "$PWRPID" 2>/dev/null
    wait "$PWRPID" 2>/dev/null || true

    t1=$(date +%s)

    # 統計と JSON 出力 (電力は tegrastats ログの平均)
    MODE="$MODE" MODE_NAME="$MODE_NAME" TAG="$tag" MODEL="$m" RES="$r" \
    ENG="$eng" VALS="$vals" PWRLOG="$pwrlog" JSON="$json" REPS="$REPS" \
    ELAPSED="$((t1-t0))" OPTSTR="$OPT" python3 - <<'PY'
import os, re, json, statistics as st

vals = [float(x) for x in os.environ["VALS"].split()]
tag  = os.environ["TAG"]

pwr = {}
try:
    txt = open(os.environ["PWRLOG"], errors="ignore").read()
    for rail in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"):
        # tegrastats 形式: "VDD_IN 5600mW/5480mW" (瞬時/平均) の瞬時側を集める
        xs = [int(v) for v in re.findall(rail + r" (\d+)mW", txt)]
        if xs:
            pwr[rail] = {
                "mean_mW": round(st.mean(xs), 1),
                "max_mW": max(xs),
                "min_mW": min(xs),
                "n": len(xs),
            }
except Exception as e:
    pwr["error"] = str(e)

out = {
    "model": os.environ["MODEL"],
    "res": int(os.environ["RES"]),
    "tag": tag,
    "engine": os.path.basename(os.environ["ENG"]),
    "engine_MB": round(os.path.getsize(os.environ["ENG"]) / 1e6, 2),
    "power_mode": os.environ["MODE_NAME"],
    "reps": int(os.environ["REPS"]),
    "n_ok": len(vals),
    "elapsed_s": int(os.environ["ELAPSED"]),
    "trt": "10.3.0",
    "l4t": "R36.4.3",
    "protocol": "loadEngine, " + os.environ["OPTSTR"].replace("--", "").replace(" ", ", ")
                + f", {os.environ['REPS']} reps",
    "power": pwr,
}
if vals:
    out.update({
        "gpu_compute_mean_ms_all_reps": vals,
        "mean_ms": round(st.mean(vals), 5),
        "sd_ms": round(st.stdev(vals), 5) if len(vals) > 1 else 0.0,
        "median_ms": round(st.median(vals), 5),
        "min_ms": min(vals), "max_ms": max(vals),
        "cv_pct": round(100 * st.stdev(vals) / st.mean(vals), 3) if len(vals) > 1 else 0.0,
    })
    if len(vals) > 1:
        out["ci95_halfwidth_ms"] = round(1.96 * st.stdev(vals) / len(vals) ** 0.5, 5)

json.dump(out, open(os.environ["JSON"], "w"), ensure_ascii=False, indent=2)
p = pwr.get("VDD_IN", {}).get("mean_mW", "-")
print(f"  -> {tag}: {out.get('mean_ms','-')} ms (n={len(vals)}/{os.environ['REPS']}) VDD_IN {p} mW")
PY

    echo "[$(date '+%H:%M:%S')] OK            ${tag}  $((t1-t0)) s"
    done_n=$((done_n+1))
    rm -f "$pwrlog"
  done
done

T_ALL1=$(date +%s)
echo
echo "=== 完了 $(date '+%F %T') mode=${MODE_NAME} ==="
echo "新規 ${done_n} / スキップ ${skip_n} / 失敗 ${fail_n}"
echo "所要 $(( (T_ALL1-T_ALL0) / 60 )) 分"
echo "JSON: $(ls ${OUT}/*.json 2>/dev/null | wc -l) 件"

touch "$E/logs/meas_trt10_${MODE}.done"
