#!/usr/bin/env bash
# 前処理レイテンシを 15W と MAXN_SUPER の両方で測り直す (CPU のみ・GPU 不使用).
#
# なぜ要るか
#   論文の e2e は optimal_n.py:92,94 が prep_timing_B.json (オフライン経路) と
#   prep_timing_deploy.json (配備経路) を GPU 時間に足して作る。この 2 つは
#   **JetPack 5.1.2 / 15W / CPU 1.51 GHz** で測った 2026-08-17 と 08-24 の値である。
#   Orin の GPU 値を TRT 10.3 へ全面差し替えると、e2e が
#   「新環境の GPU + 旧環境の CPU 前処理」の混合になり、
#   「TensorRT 版を交絡させない」という目的が前処理側の交絡に置き換わってしまう。
#   前処理は mnv4 N=224 で e2e の 40% を占めるので無視できない。
#   MAXN_SUPER では CPU が 1.51 -> 1.73 GHz になるため、両モードで測り直す。
#
# ⚠️ GPU は使わないが、CPU を占有するので **latency 測定と同時に走らせてはいけない**。
#    実行は TRT10 の全測定が終わったあと (2026-09-02 02:00 以降) にすること。
#
# ⭐ 既存の prep_timing_*.json には電力モードも CPU クロックも記録されていなかった
#    (どの環境の値かを日時から推定するしかなかった)。本スクリプトは測定のたびに
#    env メタを JSON へ注入する。
#
# usage (fgpu0):
#   bash measure_prep_mode.sh all      # 15W -> MAXN_SUPER の順に両方
#   bash measure_prep_mode.sh 15w
#   bash measure_prep_mode.sh maxn
#   FORCE=1 bash measure_prep_mode.sh all   # 既存の出力があっても測り直す
#
# 出力: results_prep_trt10/prep_timing_{A,B,deploy}_{15w,maxn}.json
set -u

# ⚠️ EDGE_ROOT は**検証用の差し替え口**である (本番では設定しない)。
#    占有中の fgpu0 で未検証のまま走らせずに済むよう、Mac のサンドボックスで
#    制御フロー (競合時に nvpmodel へ到達しないこと等) を確かめるために置いている。
EDGE="${EDGE_ROOT:-/home1/gfsi/ufsi0002/bs2026-resolution-edge}"
OUT="$EDGE/results_prep_trt10"
LOGD="$EDGE/logs"
SRC_IMAGES="$EDGE/sample_images"
CLIP="$EDGE/prep4k/clip_4k.mp4"
FRAME="$EDGE/prep4k/frame_4k.jpg"
RES_PY="$EDGE/prep_timing_res.py"
DEPLOY_PY="$EDGE/prep4k/prep_timing_deploy.py"
N=200
WARMUP=20
FORCE="${FORCE:-0}"

WHICH="${1:-all}"
case "$WHICH" in
    all|15w|maxn) ;;
    *) echo "usage: bash $0 {all|15w|maxn}"; exit 2 ;;
esac

mkdir -p "$OUT" "$LOGD"

log() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }

# ---- 資材の点検 (測定を始めてから足りないと気づくのを防ぐ) ----
miss=0
for f in "$RES_PY" "$DEPLOY_PY" "$CLIP" "$FRAME"; do
    [ -f "$f" ] || { echo "[NG] 無い: $f"; miss=1; }
done
[ -d "$SRC_IMAGES" ] || { echo "[NG] 無い: $SRC_IMAGES"; miss=1; }
nimg=$(ls "$SRC_IMAGES" 2>/dev/null | wc -l)
if [ "$nimg" -lt $((N + WARMUP)) ]; then
    echo "[NG] 画像が足りない: $nimg 枚 (要 $((N + WARMUP)))"; miss=1
fi
[ "$miss" -eq 0 ] || exit 1
log "資材 OK (画像 $nimg 枚)"

# ---- 競合チェック ----
# ⚠️ 順序が要点。**nvpmodel の切替より前**に置くこと。逆にすると、中止したときに
#    電力モードだけが変わって、走っている他の測定を壊す。
busy_reason() {
    local n
    n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
    # ⚠️ pgrep -c は該当なしのとき「0」を出力したうえで終了コード 1 を返す。
    #    `$(pgrep -c ... || echo 0)` と書くと "0\n0" の 2 行になり
    #    `[ -gt ]` が integer expression expected で失敗する (既知のバグ。持ち込まない)
    [ "$n" -gt 0 ] && { echo "trtexec が $n 本"; return 0; }
    n=$(pgrep -cf 'orin_infer' 2>/dev/null); n=${n:-0}
    [ "$n" -gt 0 ] && { echo "orin_infer が $n 本"; return 0; }
    n=$(pgrep -cf 'measure_trt10' 2>/dev/null); n=${n:-0}
    [ "$n" -gt 0 ] && { echo "measure_trt10 が $n 本"; return 0; }
    n=$(pgrep -cf 'DFobj|DetectForecast' 2>/dev/null); n=${n:-0}
    [ "$n" -gt 0 ] && { echo "検出器が $n 本"; return 0; }
    return 1
}

if reason=$(busy_reason); then
    echo "[中止] GPU/CPU を使う測定が走っている: $reason"
    echo "       前処理測定は CPU を占有するので、終わってから実行すること"
    exit 1
fi
log "競合なし"

# ---- 電力モードの切替 ----
mode_id() { case "$1" in 15w) echo 0 ;; maxn) echo 2 ;; esac; }
mode_name() { case "$1" in 15w) echo "15W" ;; maxn) echo "MAXN_SUPER" ;; esac; }

set_mode() {
    local m="$1" id
    id=$(mode_id "$m")
    log "電力モードを $(mode_name "$m") (ID=$id) へ切替"
    sudo -n nvpmodel -m "$id" >/dev/null 2>&1
    sleep 10          # クロックが落ち着くのを待つ
    local cur
    cur=$(nvpmodel -q 2>/dev/null | grep -A1 "NV Power Mode" | tail -1)
    log "  nvpmodel -q -> ${cur:-取得できず}"
    log "  CPU MaxFreq -> $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null) Hz"
}

# ---- 環境メタの注入 ----
# 既存 JSON には電力モードもクロックも入っていなかった。同じ轍を踏まない
inject_env() {
    local json="$1" m="$2"
    MODE_NAME="$(mode_name "$m")" JSON="$json" python3 - <<'PY'
import glob, json, os, platform, subprocess


def read(p, d=None):
    try:
        return open(p).read().strip()
    except Exception:
        return d


def ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


path = os.environ["JSON"]
d = json.load(open(path))
cur = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"))
d["env"] = {
    "power_mode": os.environ["MODE_NAME"],
    "nvpmodel_q": (subprocess.run(["nvpmodel", "-q"], capture_output=True,
                                  text=True).stdout or "").strip().splitlines()[-2:],
    "cpu_max_freq_Hz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
    "cpu_cur_freq_Hz": [read(p) for p in cur],
    "online_cpus": read("/sys/devices/system/cpu/online"),
    "l4t": read("/etc/nv_tegra_release", "").split(",")[0:3],
    "kernel": platform.release(),
    "python": platform.python_version(),
    "numpy": ver("numpy"), "cv2": ver("cv2"), "PIL": ver("PIL"),
    "thermal_C": {os.path.basename(os.path.dirname(p)):
                  (int(read(p, "0")) / 1000.0)
                  for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))[:6]},
    "note": ("前処理は CPU のみ (GPU 不使用)。電力モードで CPU クロックが変わるため "
             "15W と MAXN_SUPER の両方で測る"),
}
json.dump(d, open(path, "w"), indent=2, ensure_ascii=False)
print("  env 注入: %s (power_mode=%s, cpu_max=%s Hz)"
      % (os.path.basename(path), d["env"]["power_mode"], d["env"]["cpu_max_freq_Hz"]))
PY
}

# ---- 1 モードぶんの測定 ----
run_one() {
    local m="$1" tag json rc
    set_mode "$m"

    # (a) オフライン経路 mode B — 論文の prep_timing_B.json に対応
    # (b) オフライン経路 mode A — 対照 (N へ落として 224 へ戻す条件)
    for mode in B A; do
        json="$OUT/prep_timing_${mode}_${m}.json"
        if [ -s "$json" ] && [ "$FORCE" != "1" ]; then
            log "skip  prep_timing_${mode}_${m} (既存)"
        else
            log "測定 prep_timing_${mode}_${m}"
            python3 "$RES_PY" --src "$SRC_IMAGES" --mode "$mode" --n "$N" \
                --warmup "$WARMUP" --out "$json" \
                > "$LOGD/prep_${mode}_${m}.log" 2>&1
            rc=$?
            if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
                log "  [NG] rc=$rc — $LOGD/prep_${mode}_${m}.log を見よ"
                tail -5 "$LOGD/prep_${mode}_${m}.log"
                return 1
            fi
            inject_env "$json" "$m"
            log "  OK $(grep -c . "$LOGD/prep_${mode}_${m}.log") 行"
        fi
    done

    # (c) 配備経路 — 論文の prep_timing_deploy.json に対応
    json="$OUT/prep_timing_deploy_${m}.json"
    if [ -s "$json" ] && [ "$FORCE" != "1" ]; then
        log "skip  prep_timing_deploy_${m} (既存)"
    else
        log "測定 prep_timing_deploy_${m}"
        python3 "$DEPLOY_PY" --clip "$CLIP" --frame "$FRAME" --n "$N" \
            --warmup "$WARMUP" --out "$json" \
            > "$LOGD/prep_deploy_${m}.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
            log "  [NG] rc=$rc — $LOGD/prep_deploy_${m}.log を見よ"
            tail -5 "$LOGD/prep_deploy_${m}.log"
            return 1
        fi
        inject_env "$json" "$m"
        log "  OK"
    fi
    return 0
}

T0=$(date +%s)
log "===== 前処理の両モード測定 開始 (対象 $WHICH) ====="

fail=0
case "$WHICH" in
    all)  run_one 15w || fail=1
          [ $fail -eq 0 ] && { run_one maxn || fail=1; } ;;
    15w)  run_one 15w || fail=1 ;;
    maxn) run_one maxn || fail=1 ;;
esac

# ⭐ 終了時は必ず MAXN_SUPER へ戻す (JetPack 6.2 の既定・検出器 40.4 fps と同じモード)。
#    15w 単独で走らせたときも戻す
log "電力モードを MAXN_SUPER へ戻す"
sudo -n nvpmodel -m 2 >/dev/null 2>&1
sleep 5
log "  nvpmodel -q -> $(nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1)"

T1=$(date +%s)
log "===== 完了 所要 $(( (T1 - T0) / 60 )) 分 fail=$fail ====="
echo "JSON: $(ls "$OUT"/*.json 2>/dev/null | wc -l) 件"
ls -la "$OUT"/ 2>/dev/null

[ $fail -eq 0 ] && touch "$LOGD/measure_prep_mode.done"
exit $fail
