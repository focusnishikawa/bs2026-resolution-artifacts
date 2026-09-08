#!/usr/bin/env bash
# 両電力モードのアイドル時消費電力を測る.
#
# なぜ要るか: VDD_IN はボード全体の入力電力なので、測定中の値をそのまま推論時間に掛けても
# 「1 推論あたりのエネルギー」にはアイドル分が丸ごと乗る。アイドルを別に測っておけば
# **推論による増分エネルギー**も出せる (collect_powermode.py の IDLE_MW / net_energy_mJ)。
#
# ⚠️ 何も走っていないことを確認してから測ること。測定中に走らせると意味が無い。
# ⚠️ 終了時は MAXN_SUPER に戻す (論文の主モードなので、途中で落ちても既定へ復帰させる)。
#
# usage: bash idle_power.sh [秒数]   (既定 60 秒/モード)
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
SECS="${1:-60}"
OUT=results_trt10_idle.json
mkdir -p logs

# ⚠️ pgrep は該当なしのとき「0」を出力しつつ終了コード 1 を返すので代入と既定値を分ける
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then echo "[abort] trtexec が ${n} 本動作中。アイドルではない"; exit 1; fi

measure() {
    # ⚠️ `local a=$1 b=$2 c=${b}...` と 1 文にまとめてはいけない。local の引数は
    #    コマンド実行前にすべて展開されるので、${b} は「まだ未定義」として set -u に弾かれる
    #    (2026-08-31 の第 2 段でこれが起き、アイドル電力が丸ごと測れなかった)。宣言を分ける。
    local mode_id="$1" mode_name="$2"
    local log="/tmp/idle_${mode_name}.log"
    echo "=== ${mode_name} (ID=${mode_id}) のアイドル電力を ${SECS} 秒 ==="
    sudo -n nvpmodel -m "$mode_id" 2>&1 | head -2
    sleep 20                      # クロック・コア構成が落ち着くまで待つ
    ( tegrastats --interval 500 > "$log" 2>&1 ) &
    local pid=$!
    sleep "$SECS"
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true
    echo "  サンプル $(grep -c VDD_IN "$log") 件"
}

measure 2 MAXN_SUPER
measure 0 15W
echo "=== 主モード (MAXN_SUPER) へ戻す ==="
sudo -n nvpmodel -m 2 2>&1 | head -2
sleep 10
nvpmodel -q 2>&1 | head -2

SECS="$SECS" OUT="$OUT" python3 - <<'PY'
import os, re, json, statistics as st

out = {
    "note": ("アイドル時のボード電力。測定中の VDD_IN からこれを引くと推論による増分になる。"
             "tegrastats を無負荷で採取したもので、電力モードごとに測ってある"),
    "seconds_per_mode": int(os.environ["SECS"]),
    "trt": "10.3.0", "l4t": "R36.4.3",
    "modes": {},
}
for name in ("MAXN_SUPER", "15W"):
    try:
        txt = open(f"/tmp/idle_{name}.log", errors="ignore").read()
    except OSError:
        continue
    m = {}
    for rail in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"):
        xs = [int(v) for v in re.findall(rail + r" (\d+)mW", txt)]
        if xs:
            m[rail] = {"mean_mW": round(st.mean(xs), 1), "median_mW": round(st.median(xs), 1),
                       "max_mW": max(xs), "min_mW": min(xs), "n": len(xs)}
    if m:
        out["modes"][name] = m

json.dump(out, open(os.environ["OUT"], "w"), ensure_ascii=False, indent=1)
for name, m in out["modes"].items():
    v = m.get("VDD_IN", {})
    print(f"  {name:11s} VDD_IN {v.get('mean_mW','-')} mW (n={v.get('n','-')})")
print(f"[saved] {os.environ['OUT']}")
PY

touch logs/idle_power.done
