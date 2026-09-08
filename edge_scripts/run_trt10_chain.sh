#!/usr/bin/env bash
# TRT10 再ビルドの完了を待ち、15W → MAXN_SUPER の順に 56 構成を測定する。
#
# ⭐ 15W を先にするのは、最後が MAXN_SUPER で終わり **電力モードの復帰操作が不要**になるため。
# ⭐ 論文の既存値 (JP5.1.2 / TRT8.5.2) は参照しない。すべて新環境 (JP6.2 R36.4.3 / TRT10.3) で測る。
#
# 起動: nohup setsid bash run_trt10_chain.sh > logs/trt10_chain_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/trt10_chain_master.log    完了マーカー: logs/trt10_chain.done

set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
mkdir -p logs
rm -f "$E/logs/trt10_chain.done"

say() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }

say "##### TRT10 測定チェーン開始 #####"

# ---------- 1. ビルド完了を待つ ----------
if [ ! -f logs/build_trt10.done ]; then
    say "ビルド完了を待機 (60 秒ごとに確認・上限 6 時間)"
    waited=0
    while [ ! -f logs/build_trt10.done ]; do
        sleep 60
        waited=$((waited + 1))
        if [ $((waited % 10)) -eq 0 ]; then
            say "  待機 ${waited} 分 / エンジン $(ls engines_trt10/*.engine 2>/dev/null | wc -l)/56"
        fi
        # ビルドが落ちていたら待ち続けても意味がない
        if [ "$(pgrep -cf '[b]uild_trt10_cnn56' 2>/dev/null || echo 0)" -eq 0 ] && [ "$waited" -gt 2 ]; then
            if [ ! -f logs/build_trt10.done ]; then
                say "⚠️ ビルドプロセスが見当たらないのに完了マーカーが無い。中止する"
                exit 1
            fi
        fi
        if [ "$waited" -gt 360 ]; then say "⚠️ 6 時間を超えた。中止する"; exit 1; fi
    done
fi
say "ビルド完了を確認"

# ---------- 2. エンジンの員数確認 ----------
n=$(ls engines_trt10/*.engine 2>/dev/null | wc -l)
say "エンジン ${n}/56"
if [ "$n" -lt 56 ]; then
    say "⚠️ 56 個に足りない。欠けている構成は測定でスキップされる (続行する)"
fi

# ---------- 3. trtexec が残っていないか ----------
for i in $(seq 1 10); do
    c=$(pgrep -c trtexec 2>/dev/null || echo 0)
    [ "$c" -eq 0 ] && break
    say "  trtexec が ${c} 本残っている。30 秒待つ ($i/10)"
    sleep 30
done

# ---------- 4. 15W → MAXN_SUPER ----------
for mode in 15w maxn; do
    say "===== 測定開始: ${mode} ====="
    t0=$(date +%s)
    bash measure_trt10_mode.sh "$mode"
    rc=$?
    t1=$(date +%s)
    say "===== 測定終了: ${mode} rc=${rc} $(( (t1-t0)/60 )) 分 ====="
    if [ "$rc" -ne 0 ]; then
        say "⚠️ ${mode} が rc=${rc} で終了した。チェーンを中止する"
        exit "$rc"
    fi
    sleep 20
done

# ---------- 5. 後始末 ----------
say "--- 最終状態 ---"
nvpmodel -q 2>&1 | head -2
say "15w JSON: $(ls results_trt10_15w/*.json 2>/dev/null | wc -l) 件"
say "maxn JSON: $(ls results_trt10_maxn/*.json 2>/dev/null | wc -l) 件"
say "##### チェーン完了 #####"

touch "$E/logs/trt10_chain.done"
