#!/usr/bin/env bash
# 判別 1 回あたりの **連続実行 wall-clock** を測る (fgpu0・査読指摘 A-6b).
#
# 論文の判別時間は「前処理 (別測定) + 各パートの trtexec GPU 時間の**和**」である。
# ViT-L は 5 段 (DINOv2-L) / 4 段 (DINOv3-L) に分割してあるため、和にはホスト側の
# 呼び出し・同期・段間の受け渡しが入らない。**10 ms 予算で選ばれる DINOv2-L N=64 は
# 合算 9.402 ms で余裕が 0.6 ms しかない**ので、合算モデルが実行時間として通用するかを
# 実機で確かめる必要がある (再レビュー A-6b)。
#
# 測るもの: 1 枚あたり H2D コピー -> 全段 enqueue -> D2H コピー -> 同期 の wall-clock。
#   ⚠️ CSV 整形は配備時のコストではないので計時に含めない (プログラム側で除外済み)。
#   ⚠️ CPU 前処理はここに含まない (別測定の prep_deploy を足して e2e とする)。
#
# ⛔ **排他実行が必須**。他に trtexec や推論が走っていると値が汚れる
#    ([[feedback_fgpu0_exclusive_benchmark]] : 2026-07-10 に 1 本汚染した実例あり)。
#
# usage: bash measure_wallclock_trt10.sh [出力JSON]
set -u
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
OUT="${1:-results_wallclock_trt10.json}"
BIN=./orin_infer_chain_trt10_timed
N="${N:-1882}"
WARMUP="${WARMUP:-200}"
REP="${REP:-3}"        # 構成あたりの反復 (別プロセス起動)

[ -x "$BIN" ] || { echo "[abort] $BIN が無い (先に build すること)"; exit 1; }

# ---- 競合チェック。⚠️ pgrep -f は自分のコマンドラインにも一致するので使わない ----
for p in trtexec orin_infer_chain orin_infer_chain_trt10; do
    n=$(pgrep -c "$p" 2>/dev/null); n=${n:-0}
    [ "$n" -gt 0 ] && { echo "[abort] $p が ${n} 本走っている。排他で測れないので中止"; exit 1; }
done
echo "電力モード: $(sudo nvpmodel -q 2>/dev/null | grep -i 'power mode' | head -1)"

mkdir -p logs/wallclock results_wallclock
TMPCSV=/tmp/wallclock_out.csv

# run <label> <入力bin> <engine...>
run() {
    local label="$1" bin="$2"; shift 2
    local e
    for e in "$@"; do [ -s "$e" ] || { echo "  [miss] $label ($e が無い)"; return 1; }; done
    [ -s "$bin" ] || { echo "  [miss] $label (入力 $bin が無い)"; return 1; }
    local r
    for r in $(seq 1 "$REP"); do
        WARMUP="$WARMUP" TIMES_OUT="results_wallclock/${label}_rep${r}.txt" \
            "$BIN" "$TMPCSV" "$N" "$bin" "$@" > "logs/wallclock/${label}_rep${r}.log" 2>&1
        local rc=$?
        local line; line=$(grep -a "^\[wall\]" "logs/wallclock/${label}_rep${r}.log" | tail -1)
        printf "  %-22s rep%d rc=%d %s\n" "$label" "$r" "$rc" "${line:-(計時行なし)}"
    done
}

echo "===== 連続実行 wall-clock 測定 開始 $(date '+%F %T') ====="
echo "  枚数 ${N} / ウォームアップ ${WARMUP} / 反復 ${REP}"

# (1) 10 ms 予算で選ばれる 2 構成 (前処理経路で選び分かれる当事者)
run "resnet50_r224"  "inputs/inputs_r224.bin"            "engines_trt10/resnet50_r224.engine"
run "dinov2_l_r64"   "inputs/real_dinov2_l_r64.fp16.bin" engines_trt10_split/dinov2_l_r64_{p0,p1,p2,p3s0,p3s1}.engine

# (2) 代表的な高解像度 ViT-L (段数が同じでも 1 段が重い側の確認)
run "dinov2_l_r224"  "inputs/real_dinov2_l_r224.fp16.bin" engines_trt10_split/dinov2_l_r224_{p0,p1,p2,p3s0,p3s1}.engine
run "dinov3_l_r224"  "inputs/real_dinov3_l_r224.fp16.bin" engines_trt10_split/dinov3_l_r224_p{0,1,2,3}.engine

# (3) 対照: 単段の小型モデル (段間コストが無い側の基準)
run "resnet50_r112"  "inputs/inputs_r112.bin"            "engines_trt10/resnet50_r112.engine"

echo "===== 終了 $(date '+%F %T') ====="
echo "  生の計時: $E/results_wallclock/*.txt"
echo "  ログ    : $E/logs/wallclock/*.log"
