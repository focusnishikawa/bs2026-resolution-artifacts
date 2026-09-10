#!/usr/bin/env bash
# ViT-S/16 FP32 実測を頭から尻まで 1 本で回すオーケストレータ。
#
#   Phase L0  FP32 エンジン構築       build_trt10_fp32_vits.sh        (14 本)
#   Phase L1  レイテンシ MAXN_SUPER   measure_trt10_fp32_mode.sh maxn (14 JSON)
#   Phase L2  レイテンシ 15W          measure_trt10_fp32_mode.sh 15w  (14 JSON)
#   Phase L3  電力モードを MAXN_SUPER へ復帰
#   Phase A   30 シード配備精度       run_30seed_deploy_acc_fp32.sh 42 71 (test+val)
#
# ⭐ L2 (15W) の直後に L3 を挟むのは、Phase A を **15W のまま走らせないため**である。
#    Phase A は数十時間かかるので、電力モードの取り違えに気づくのが遅れると全部やり直しになる。
# ⭐ どこかのフェーズが非ゼロで終わったら、そこで止めて **どのフェーズで止まったか**を出す。
#    後続を巻き添えにしない (壊れたエンジンで精度を測ると気づけない)。
# ⭐ 各サブスクリプトは冪等なので、直したうえで同じコマンドを再投入すれば途中から再開できる。
#
# 起動: nohup setsid bash run_vits_fp32_all.sh > logs/vits_fp32_all_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/vits_fp32_all_master.log     完了マーカー: logs/vits_fp32_all.done

set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
mkdir -p logs
rm -f "$E/logs/vits_fp32_all.done"

N_EXPECT=14                   # ViT-S/16 x 14 解像度
S0=42; S1=71                  # FP32 は seed42 も未測定なので 30 シード

say() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }

PH_NAME=(); PH_SEC=(); PH_RC=()

summary() {
    echo
    echo "----- フェーズ別 経過時間 -----"
    i=0
    while [ "$i" -lt "${#PH_NAME[@]}" ]; do
        printf "  %-28s rc=%-3s %5d 分\n" "${PH_NAME[$i]}" "${PH_RC[$i]}" "$(( PH_SEC[i] / 60 ))"
        i=$((i + 1))
    done
    echo "  合計 $(( ($(date +%s) - T_ALL0) / 60 )) 分 ($(( ($(date +%s) - T_ALL0) / 3600 )) 時間)"
}

# フェーズを 1 つ走らせる。第 1 引数=表示名、第 2 引数=期待する done マーカー ("" なら見ない)、
# 残り=実行するコマンド。非ゼロ終了か done マーカー欠落で 1 を返す。
run_phase() {
    local name="$1" marker="$2"; shift 2
    echo
    say "########## ${name} 開始  $(date '+%F %T') ##########"
    say "  cmd: $*"
    local t0 t1 rc
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    PH_NAME+=("$name"); PH_SEC+=("$((t1 - t0))"); PH_RC+=("$rc")
    say "########## ${name} 終了 rc=${rc}  $(( (t1 - t0) / 60 )) 分  $(date '+%F %T') ##########"
    if [ "$rc" -ne 0 ]; then
        return 1
    fi
    if [ -n "$marker" ] && [ ! -f "$marker" ]; then
        say "⚠️ ${name} は rc=0 だが完了マーカー ${marker} が無い"
        return 1
    fi
    return 0
}

# 直前のフェーズの trtexec が残っていないか確認する (残っていると次のフェーズが
# 競合チェックで abort してしまう)。run_trt10_chain.sh と同じ待ち方である。
wait_trtexec_clear() {
    local i c
    for i in $(seq 1 10); do
        c=$(pgrep -c trtexec 2>/dev/null || true); c=${c:-0}
        [ "$c" -eq 0 ] && return 0
        say "  trtexec が ${c} 本残っている。30 秒待つ ($i/10)"
        sleep 30
    done
    return 0
}

# 止まったフェーズを明示して終了する。
die_at() {
    echo
    say "⛔⛔ $1 で停止した。後続フェーズは実行しない ⛔⛔"
    say "    ログ: logs/vits_fp32_all_master.log ほか各フェーズのログを確認すること"
    summary
    exit 1
}

T_ALL0=$(date +%s)
say "##### ViT-S/16 FP32 実測チェーン開始 #####"
say "  base=${E}  seeds ${S0}-${S1}  期待構成数 ${N_EXPECT}"

# ---------- 競合チェック ----------
n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then
    say "[abort] trtexec が ${n} 本動作中。別のビルドか測定が走っている"
    exit 1
fi

# ---------- Phase L0: FP32 エンジン構築 ----------
run_phase "Phase L0 (build fp32)" "logs/build_trt10_fp32_vits.done" \
          bash build_trt10_fp32_vits.sh \
    || die_at "Phase L0 (build fp32)"

n_eng=$(ls engines_trt10_fp32/*.engine 2>/dev/null | wc -l)
say "  FP32 エンジン ${n_eng}/${N_EXPECT}"
if [ "$n_eng" -lt "$N_EXPECT" ]; then
    say "⚠️ エンジンが足りない (失敗が $(( N_EXPECT - n_eng )) 本)。測定に進まない"
    die_at "Phase L0 (build fp32)"
fi

# ---------- Phase L1: レイテンシ MAXN_SUPER ----------
wait_trtexec_clear
run_phase "Phase L1 (latency maxn)" "logs/meas_trt10_fp32_maxn.done" \
          bash measure_trt10_fp32_mode.sh maxn \
    || die_at "Phase L1 (latency maxn)"
say "  maxn JSON: $(ls results_trt10_fp32_maxn/*.json 2>/dev/null | wc -l)/${N_EXPECT} 件"
sleep 20

# ---------- Phase L2: レイテンシ 15W ----------
wait_trtexec_clear
run_phase "Phase L2 (latency 15w)" "logs/meas_trt10_fp32_15w.done" \
          bash measure_trt10_fp32_mode.sh 15w \
    || die_at "Phase L2 (latency 15w)"
say "  15w JSON: $(ls results_trt10_fp32_15w/*.json 2>/dev/null | wc -l)/${N_EXPECT} 件"
sleep 20

# ---------- Phase L3: 電力モードを MAXN_SUPER へ復帰 ----------
echo
say "########## Phase L3 (restore MAXN_SUPER) 開始  $(date '+%F %T') ##########"
t0=$(date +%s)
sudo -n nvpmodel -m 2 2>&1 | head -3
rc=${PIPESTATUS[0]}           # ⚠️ パイプの後の $? は head の結果。nvpmodel 側を見る
sleep 15                      # クロック・コア構成が落ち着くのを待つ
say "--- 復帰後の状態 ---"
nvpmodel -q 2>&1 | head -2
say "Online CPUs: $(nproc) / MaxFreq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)"
t1=$(date +%s)
PH_NAME+=("Phase L3 (restore MAXN)"); PH_SEC+=("$((t1 - t0))"); PH_RC+=("$rc")
say "########## Phase L3 終了 rc=${rc}  $(date '+%F %T') ##########"
[ "$rc" -ne 0 ] && die_at "Phase L3 (restore MAXN)"

# ---------- Phase A: 30 シード配備精度 (test+val) ----------
wait_trtexec_clear
run_phase "Phase A (30seed acc fp32)" "logs/deploy_acc_30seed_fp32.done" \
          bash run_30seed_deploy_acc_fp32.sh "$S0" "$S1" \
    || die_at "Phase A (30seed acc fp32)"

# ---------- 最終確認 ----------
echo
say "===== 成果物 ====="
say "  engines_trt10_fp32   : $(ls engines_trt10_fp32/*.engine 2>/dev/null | wc -l)/${N_EXPECT} 本"
say "  results_trt10_fp32_maxn: $(ls results_trt10_fp32_maxn/*.json 2>/dev/null | wc -l)/${N_EXPECT} 件"
say "  results_trt10_fp32_15w : $(ls results_trt10_fp32_15w/*.json 2>/dev/null | wc -l)/${N_EXPECT} 件"
say "  preds_30seed_fp32      : $(find preds_30seed_fp32     -name '*.csv' 2>/dev/null | wc -l) 件 (期待 $(( (S1 - S0 + 1) * N_EXPECT )))"
say "  preds_30seed_fp32_val  : $(find preds_30seed_fp32_val -name '*.csv' 2>/dev/null | wc -l) 件 (期待 $(( (S1 - S0 + 1) * N_EXPECT )))"
say "--- 最終電力モード ---"
nvpmodel -q 2>&1 | head -2

summary
echo
say "##### ViT-S/16 FP32 実測チェーン完了 $(date '+%F %T') #####"

touch "$E/logs/vits_fp32_all.done"
exit 0
