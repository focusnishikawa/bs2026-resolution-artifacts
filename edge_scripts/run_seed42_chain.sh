#!/usr/bin/env bash
# 第 3 段: seed 42 を測り、Orin の配備精度を 30 シード (42-71) へそろえる.
#
# 背景:
#   論文はサーバ FP32 の精度を **30 シード (42-71)** で報告している。一方 Orin の配備精度は
#   onnx_seeds/ に s43-s71 の **29 シード**しか無く 1 本足りなかった (test / val とも)。
#   seed 42 の ONNX は `onnx/` にある。**models_s42 由来であることは独立の証拠で確定済み**:
#     - docs/bs2026-resolution_報告書.md:867  export_onnx.py --models_dir models_s42
#     - ⭐ 決め手は **FP32 同士の照合** (半精度の影響が混ざらない比較)。Orin の FP32 エンジン
#       vit_small_r112 との argmax 一致率は base 90.54% (不一致 178 枚) に対し
#       **s42 は 99.95% (不一致 1 枚 = 数値誤差)**  (train/fp16_accuracy_trt10.py:36-42)
#   そこで `onnx_seeds/s42 -> ../onnx` の symlink を張り、他シードと**同じ経路**で測る。
#
# ⚠️ 実行中の run_30seed_chain_v2.sh とは競合させない。第 2 段の完了マーカーを待ってから動く
#    (精度測定なので競合しても値は壊れないが、双方が遅くなるだけで得が無い)。
# ⚠️ **チェーンのマーカー deploy_acc_30seed_v2_all.done は消さない**。
#    Mac 側 (train/refresh_a1.sh) の完了判定はマーカーではなく
#    「test / val の両方で 30 シード x 56 CSV がそろっているか」というデータ側の実測で行う。
#
# 使い方 (fgpu0 上):
#   nohup setsid bash run_seed42_chain.sh > logs/seed42_master.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")" || exit 1

M=logs/deploy_acc_30seed_v2_all.done
LIMIT=$((24 * 60 * 60))          # 24 時間待っても来なければ諦める (無言で待ち続けない)
WAITED=0

echo "===== 第 3 段 (seed 42) 待機開始 $(date '+%F %T') ====="
echo "  第 2 段の完了マーカー ($M) を待つ"
while [ ! -f "$M" ]; do
    sleep 120
    WAITED=$((WAITED + 120))
    if [ "$WAITED" -ge "$LIMIT" ]; then
        echo "⛔ 24 時間待ってもマーカーが来ない。チェーンが落ちた可能性がある。中止する"
        echo "   確認: tail logs/deploy_acc_30seed_v2_master.log"
        exit 1
    fi
    if [ $((WAITED % 3600)) -eq 0 ]; then
        echo "  [$(date '+%H:%M:%S')] 待機中 ($((WAITED / 3600)) 時間経過)"
    fi
done

echo "===== 第 2 段の完了を検知 $(date '+%F %T') ====="

# ONNX の実在を先に確かめる (無いと run_30seed_deploy_acc_v2.sh は黙ってスキップする)
n=0
for m in mnv4 effb0 resnet50 vit_small; do
    for r in 16 32 48 64 80 96 112 128 144 160 176 192 208 224; do
        [ -f "onnx_seeds/s42/${m}_r${r}.onnx" ] && n=$((n + 1))
    done
done
echo "  onnx_seeds/s42 の解決数: $n / 56"
if [ "$n" -ne 56 ]; then
    echo "⛔ seed 42 の ONNX がそろっていない。中止する"
    echo "   復旧: ln -s ../onnx onnx_seeds/s42"
    exit 1
fi

echo
echo "########## seed 42 (test+val) $(date '+%F %T') ##########"
bash run_30seed_deploy_acc_v2.sh 42 42
rc=$?
echo "########## seed 42 終了 rc=$rc $(date '+%F %T') ##########"

echo
echo "===== 第 3 段 完了 $(date '+%F %T') ====="
echo "test CSV: $(find preds_30seed     -name '*.csv' | wc -l) 件 (期待 1680 = 30 seed x 56)"
echo "val  CSV: $(find preds_30seed_val -name '*.csv' | wc -l) 件 (期待 1680)"

# ⭐ 由来の実地検証: seed 42 の test は S9/S10 の全数推論 (preds_trt10/) が既にある。
#    同じ ONNX から作ったエンジンなら argmax はほぼ一致するはずで、
#    一致しなければ onnx_seeds/s42 の指し先が seed 42 でないということになる。
echo
echo "===== 由来の検証 (新 preds_30seed/s42 対 既存 preds_trt10) ====="
for m in mnv4 effb0 resnet50 vit_small; do
    for r in 16 112 224; do
        a="preds_30seed/s42/${m}_r${r}.csv"; b="preds_trt10/${m}_r${r}.csv"
        if [ -f "$a" ] && [ -f "$b" ]; then
            same=$(paste -d, <(cut -d, -f8 "$a") <(cut -d, -f8 "$b") \
                   | awk -F, 'NR>1{t++; if($1==$2) s++} END{if(t) printf "%.2f%% (%d/%d)", 100*s/t, s, t}')
            printf "  %-14s %s\n" "${m}_r${r}" "$same"
        else
            printf "  %-14s 片方が無い (a=%s b=%s)\n" "${m}_r${r}" \
                   "$([ -f "$a" ] && echo あり || echo なし)" "$([ -f "$b" ] && echo あり || echo なし)"
        fi
    done
done

[ "$rc" -eq 0 ] && touch logs/deploy_acc_30seed_v2_s42.done
exit 0
