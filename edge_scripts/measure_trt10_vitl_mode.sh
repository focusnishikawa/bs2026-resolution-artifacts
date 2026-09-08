#!/usr/bin/env bash
# ViT-L の分割チェーンを TensorRT 10.3 エンジンで測定する。電力モードごとに実行する。
#
# measure_trt10_mode.sh (CNN 56 構成) の ViT-L 版。プロトコルと JSON 形式を揃えてあるので
# collect_powermode.py と同じ集計コードで扱える。
#
# ⭐ プロトコルは既存 measure_all30.sh の ViT-L 部を踏襲:
#    trtexec --loadEngine --warmUp=2000 --iterations=200 --avgRuns=50 を
#    **30 回別プロセスで**起動する (CNN は iterations=300/avgRuns=100 で、ViT-L だけ軽い。
#    1 パートが数 ms〜数十 ms と重いため。既存値と揃えるためここは変えない)。
#
# ⭐ 消費電力 (VDD_IN / VDD_CPU_GPU_CV / VDD_SOC) を同時記録する。
#
# 測る対象:
#   deploy (既定) : 配備構成 126 パート = DINOv2-L 14x5 + DINOv3-L 14x4
#   ctrl          : 全段 FP32 対照 24 + ViT-S/16 FP32 2 + 単体エンジン 2
#                   (電力モード比較には要らないので MAXN でだけ測ればよい)
#
# usage:
#   bash measure_trt10_vitl_mode.sh maxn [deploy|ctrl|all]
#   bash measure_trt10_vitl_mode.sh 15w  [deploy|ctrl|all]
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1

MODE="${1:-}"
WHAT="${2:-deploy}"
case "$MODE" in
    maxn) MODE_ID=2; MODE_NAME="MAXN_SUPER" ;;
    15w)  MODE_ID=0; MODE_NAME="15W" ;;
    *) echo "使い方: bash $0 {maxn|15w} [deploy|ctrl|all]"; exit 1 ;;
esac
case "$WHAT" in deploy|ctrl|all) ;; *) echo "第2引数は deploy|ctrl|all"; exit 1 ;; esac

OUT="results_trt10_vitl_${MODE}"
ENGDIR=engines_trt10_split
TRTEXEC=/usr/src/tensorrt/bin/trtexec
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
CTRL_RES="16 32 64 112 128 224"
REPS=30
OPT="--warmUp=2000 --iterations=200 --avgRuns=50"

# ⭐ 測定から外すモデル (2026-09-01 追加。空白区切り・既定は空 = 従来どおり全部測る)
#    2026-09-01 の全数推論で **DINOv3-L が全 14 解像度で定数出力 (相異なる出力 1/1882)** と
#    判明した。latency も精度も使えないので、測定時間を無駄にしないための口である。
#      SKIP_MODELS="dinov3_l" bash measure_trt10_vitl_mode.sh 15w all
#    ⚠️ 外したパートは JSON が作られないので、collect_vitl_edge.py が missing_parts として
#       警告する (黙って短い合計を出すことはない)。DINOv3-L が直ったら指定を外して再実行すれば、
#       既存の DINOv2-L 分はスキップされ DINOv3-L だけが測られる。
SKIP_MODELS="${SKIP_MODELS:-}"

skip_model() {
    [ -n "$SKIP_MODELS" ] || return 1
    case " $SKIP_MODELS " in *" $1 "*) return 0 ;; esac
    return 1
}

mkdir -p "$OUT" logs
rm -f "logs/meas_trt10_vitl_${MODE}_${WHAT}.done"

# ---- 競合チェック ----
# ⚠️ pgrep は該当なしのとき「0」を出力しつつ終了コード 1 を返すので、
#    `$(pgrep -c ... || echo 0)` は 2 行になって比較が壊れる (既知バグ)。代入と既定値を分ける。
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then echo "[abort] trtexec が ${n} 本動作中。ビルドか別の測定が走っている"; exit 1; fi

# ---- 電力モードを設定 ----
echo "=== 電力モードを ${MODE_NAME} (ID=${MODE_ID}) に設定 ==="
sudo -n nvpmodel -m "$MODE_ID" 2>&1 | head -3
sleep 15
nvpmodel -q 2>&1 | head -2
echo "Online CPUs: $(nproc) / MaxFreq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)"
echo

T_ALL0=$(date +%s); done_n=0; skip_n=0; fail_n=0

# measure_one <tag> <model> <res> <part> <engine>
measure_one() {
    local tag="$1" model="$2" res="$3" part="$4" eng="$5"
    local json="${OUT}/${tag}.json"
    # ⭐ 呼び出し側ではなくここで弾く (deploy ループと ctrl の single の両方に効かせるため)
    if skip_model "$model"; then
        echo "[$(date '+%H:%M:%S')] SKIP(model)    ${tag}"; skip_n=$((skip_n+1)); return
    fi
    if [ ! -s "$eng" ]; then
        echo "[$(date '+%H:%M:%S')] MISSING engine ${tag}"; fail_n=$((fail_n+1)); return
    fi
    if [ -s "$json" ]; then
        echo "[$(date '+%H:%M:%S')] skip           ${tag} (既存)"; skip_n=$((skip_n+1)); return
    fi
    local t0 t1 pwrlog vals v replog PWRPID
    t0=$(date +%s)
    pwrlog="/tmp/pwr_vitl_${MODE}_${tag}.log"
    ( tegrastats --interval 500 > "$pwrlog" 2>&1 ) &
    PWRPID=$!
    vals=""
    for rep in $(seq 1 "$REPS"); do
        replog="/tmp/rep_vitl_${MODE}_${tag}.log"
        if timeout 600 "$TRTEXEC" --loadEngine="$eng" $OPT > "$replog" 2>&1; then
            v=$(grep -aoE "GPU Compute Time:.*mean = [0-9.]+" "$replog" | tail -1 | grep -oE "[0-9.]+$")
            [ -n "$v" ] && vals="${vals} ${v}"
        fi
    done
    kill "$PWRPID" 2>/dev/null
    # ⚠️ 引数なしの wait は使わない (監視プロセスまで待って止まる)
    wait "$PWRPID" 2>/dev/null || true
    t1=$(date +%s)

    MODE_NAME="$MODE_NAME" TAG="$tag" MODEL="$model" RES="$res" PART="$part" \
    ENG="$eng" VALS="$vals" PWRLOG="$pwrlog" JSON="$json" REPS="$REPS" \
    ELAPSED="$((t1-t0))" OPTSTR="$OPT" python3 - <<'PY'
import os, re, json, statistics as st

vals = [float(x) for x in os.environ["VALS"].split()]
tag = os.environ["TAG"]

pwr = {}
try:
    txt = open(os.environ["PWRLOG"], errors="ignore").read()
    for rail in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"):
        xs = [int(v) for v in re.findall(rail + r" (\d+)mW", txt)]
        if xs:
            pwr[rail] = {"mean_mW": round(st.mean(xs), 1), "max_mW": max(xs),
                         "min_mW": min(xs), "n": len(xs)}
except Exception as e:
    pwr["error"] = str(e)

out = {
    "model": os.environ["MODEL"],
    "res": int(os.environ["RES"]),
    "part": os.environ["PART"],
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
print(f"  -> {tag}: {out.get('median_ms','-')} ms (n={len(vals)}/{os.environ['REPS']}) VDD_IN {p} mW")
PY

    echo "[$(date '+%H:%M:%S')] OK             ${tag}  $((t1-t0)) s"
    done_n=$((done_n+1))
    rm -f "$pwrlog"
}

if [ "$WHAT" = "deploy" ] || [ "$WHAT" = "all" ]; then
    echo "===== 配備構成 126 パート x ${REPS} 回 (${MODE_NAME}) ====="
    for r in $RES; do
        echo "--- N=${r} ---"
        for part in p0 p1 p2 p3s0 p3s1; do
            measure_one "dinov2_l_r${r}_${part}" dinov2_l "$r" "$part" \
                        "$ENGDIR/dinov2_l_r${r}_${part}.engine"
        done
        for k in 0 1 2 3; do
            measure_one "dinov3_l_r${r}_p${k}" dinov3_l "$r" "p${k}" \
                        "$ENGDIR/dinov3_l_r${r}_p${k}.engine"
        done
    done
fi

if [ "$WHAT" = "ctrl" ] || [ "$WHAT" = "all" ]; then
    echo "===== 対照・参照 (${MODE_NAME}) ====="
    for r in $CTRL_RES; do
        for part in p0 p1 p2 p3; do
            measure_one "dinov2_l_r${r}_f32_${part}" dinov2_l_f32 "$r" "$part" \
                        "$ENGDIR/dinov2_l_r${r}_f32_${part}.engine"
        done
    done
    for r in 112 224; do
        measure_one "vit_small_r${r}_fp32" vit_small_fp32 "$r" whole \
                    "$ENGDIR/vit_small_r${r}_fp32.engine"
    done
    for tag in dinov2_l_r32_single dinov3_l_r32_single; do
        measure_one "$tag" "${tag%%_r32_single}" 32 single "$ENGDIR/${tag}.engine"
    done
fi

T_ALL1=$(date +%s)
echo
echo "=== 完了 $(date '+%F %T') mode=${MODE_NAME} what=${WHAT} ==="
echo "新規 ${done_n} / スキップ ${skip_n} / 失敗 ${fail_n}"
echo "所要 $(( (T_ALL1-T_ALL0) / 60 )) 分"
echo "JSON: $(ls ${OUT}/*.json 2>/dev/null | wc -l) 件"
touch "logs/meas_trt10_vitl_${MODE}_${WHAT}.done"
