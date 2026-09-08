#!/usr/bin/env bash
# 第 2 段チェーン: ViT-L を TRT 10.3 で作り直し、全数推論と両モード測定まで無人で通す.
#
# 第 1 段 (run_trt10_chain.sh) は CNN 56 構成の 15W/MAXN 測定。本スクリプトはその完了を
# 待ってから走り、論文の Orin 値を全て新環境へ差し替えるのに必要な残り全部を埋める。
#
# 段取り (この順序である理由も書いておく):
#   1. バイナリのコンパイルとスモーク  ... ビルドより先に済ませ、問題があれば
#                                          8 時間のビルド中に直せるようにする
#   2. アイドル電力 (MAXN -> 15W -> MAXN) ... 何も走っていない今しか測れない
#   3. ViT-L 154 本を再ビルド (MAXN)    ... 速いモードで作る。ビルドは測定ではないので
#                                          モードが結果に影響しない
#   4. 全数推論 (MAXN)                  ... 定数出力エンジンをここで必ず弾く
#   5. 15W で 126 パート x 30 回
#   6. MAXN で 126 パート x 30 回 + 対照/参照
#      -> 15W を先にするのは、終了時に主モード (MAXN_SUPER) で止まるようにするため
#
# ⚠️ pgrep は該当なしのとき「0」を出力しつつ終了コード 1 を返す。
#    `$(pgrep -c ... || echo 0)` は "0\n0" になり比較が壊れる (run_trt10_chain.sh の既知バグ)。
#    本スクリプトでは代入と既定値を必ず分けてある。
# ⚠️ 引数なしの `wait` は使わない (監視プロセスまで待って止まる)。
#
# 各段は既存成果物をスキップするので、落ちても同じコマンドで再開できる。
#
# 起動:
#   nohup setsid bash run_trt10_stage2_chain.sh > logs/trt10_stage2_master.log 2>&1 < /dev/null &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
mkdir -p logs
rm -f logs/trt10_stage2.done

say() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }

say "===== 第 2 段チェーン開始 ====="

# ---------- 0. 第 1 段 (CNN 56 構成 x 2 モード) の完了を待つ ----------
if [ ! -f logs/trt10_chain.done ]; then
    say "第 1 段の完了を待機 (logs/trt10_chain.done)"
    waited=0
    while [ ! -f logs/trt10_chain.done ]; do
        sleep 60
        waited=$((waited + 1))
        if [ $((waited % 30)) -eq 0 ]; then
            say "  待機 ${waited} 分 / 15w=$(ls results_trt10_15w/*.json 2>/dev/null | wc -l)/56 maxn=$(ls results_trt10_maxn/*.json 2>/dev/null | wc -l)/56"
        fi
        # 12 時間待っても終わらなければ異常とみなす (通常は約 5 時間)
        if [ "$waited" -gt 720 ]; then say "[abort] 第 1 段が 12 時間経っても終わらない"; exit 1; fi
    done
    say "第 1 段の完了を確認"
fi

# 念のため trtexec が残っていないか見る
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then
    say "trtexec が ${n} 本残っている。10 分待つ"
    sleep 600
    n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then say "[abort] trtexec が ${n} 本動作中"; exit 1; fi
fi

# ---------- 1. バイナリのコンパイルとスモーク ----------
say "----- 1. 推論バイナリ (TRT10 移植版) をコンパイル -----"
for f in orin_infer_chain orin_infer_res; do
    if [ -x "${f}_trt10" ]; then
        say "  ${f}_trt10 は既存"
    else
        g++ "${f}_trt10.cpp" -O2 -std=c++14 \
            -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include \
            -L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64 \
            -lnvinfer -lcudart -o "${f}_trt10" 2> "logs/build_${f}_trt10.log"
        say "  ${f}_trt10 rc=$?"
    fi
done
# スモークは CNN エンジンだけで行う (ViT-L はまだ無い)。失敗しても止めない
bash infer_trt10_all.sh smoke 2>&1 | sed 's/^/    /'

# ---------- 2. アイドル電力 ----------
say "----- 2. アイドル電力 (両モード) -----"
if [ -s results_trt10_idle.json ]; then
    say "  既存のためスキップ"
else
    bash idle_power.sh 60 2>&1 | sed 's/^/    /'
fi

# ---------- 3. ViT-L 再ビルド ----------
say "----- 3. ViT-L 154 本を TRT 10.3 で再ビルド (見込み 約 10 時間) -----"
if [ -f logs/build_trt10_vitl.done ]; then
    say "  完了マーカーがあるのでスキップ"
else
    bash build_trt10_vitl.sh build 2>&1 | tail -40
    say "  ビルド後のエンジン: $(ls engines_trt10_split/*.engine 2>/dev/null | wc -l) 本"
fi

# ---------- 4. 全数推論 ----------
say "----- 4. 全数推論 1,882 枚 (見込み 約 1.5 時間) -----"
if [ -f logs/infer_trt10.done ]; then
    say "  完了マーカーがあるのでスキップ"
else
    bash infer_trt10_all.sh all 2>&1 | tail -60
fi

# ---------- 5. 15W 測定 ----------
say "----- 5. 15W で ViT-L 126 パート x 30 回 (見込み 約 7 時間) -----"
bash measure_trt10_vitl_mode.sh 15w deploy > logs/meas_trt10_vitl_15w.log 2>&1
say "  15W 完了: $(ls results_trt10_vitl_15w/*.json 2>/dev/null | wc -l)/126"

# ---------- 6. MAXN 測定 (主モード) ----------
say "----- 6. MAXN_SUPER で ViT-L 126 パート x 30 回 (見込み 約 7 時間) -----"
bash measure_trt10_vitl_mode.sh maxn deploy > logs/meas_trt10_vitl_maxn.log 2>&1
say "  MAXN 配備構成 完了: $(ls results_trt10_vitl_maxn/*.json 2>/dev/null | wc -l)/126"

say "----- 6b. MAXN で対照・参照 (見込み 約 1.5 時間) -----"
bash measure_trt10_vitl_mode.sh maxn ctrl > logs/meas_trt10_vitl_maxn_ctrl.log 2>&1
say "  MAXN 全体: $(ls results_trt10_vitl_maxn/*.json 2>/dev/null | wc -l) 件"

# ---------- 終了処理 ----------
say "----- 電力モードを主モード (MAXN_SUPER) に確定 -----"
sudo -n nvpmodel -m 2 2>&1 | head -2
sleep 5
nvpmodel -q 2>&1 | head -2

touch logs/trt10_stage2.done
say "===== 第 2 段チェーン完了 ====="
