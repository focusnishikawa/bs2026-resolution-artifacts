#!/usr/bin/env bash
# Orin の残り工程を 1 本に連結する (人手の投入を挟まずに最後まで走らせるため).
#
# 経緯: 工程を段ごとに手で投入すると、投入する側 (作業セッション) が切れた時点で
#       残りが止まる。Orin は排他なので並列にもできない。そこで
#         1. 実行中の CNN 30 回測定の完了を待つ
#         2. DINOv2-L 全段 FP32 のビルドと全数推論 (図 6 の対照)
#         3. ViT-L 126 パートの 30 回測定
#       を 1 本のデタッチしたプロセスに閉じ込める。
#
# 各段は既存成果物をスキップするので、途中で落ちても同じコマンドで再開できる。
# 待機はマーカーファイル (logs/meas30_cnn.done) を見るだけで、走っているジョブには触らない。
#
# 起動 (Orin): nohup setsid bash run_orin_all30.sh > logs/orin_all30_master.log 2>&1 < /dev/null &
# 監視:        tail -f logs/orin_all30_master.log   完了マーカー logs/orin_all30.done
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
rm -f "$E/logs/orin_all30.done"

echo "[$(date '+%m/%d %H:%M:%S')] ##### Orin 残り工程 開始 #####"

# ---- 1. 実行中の CNN 30 回測定の完了を待つ ----
if [ ! -f "$E/logs/meas30_cnn.done" ]; then
    echo "[$(date '+%m/%d %H:%M:%S')] CNN 30 回測定の完了を待機中 (60 秒ごとに確認)"
    waited=0
    while [ ! -f "$E/logs/meas30_cnn.done" ]; do
        sleep 60
        waited=$((waited + 1))
        if [ $((waited % 15)) -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')]   待機 ${waited} 分 / CNN $(ls "$E"/results_v30/*.json 2>/dev/null | wc -l)/56"
        fi
        # 走っているはずの測定が消えていたら、待ち続けても意味がないので抜ける
        if [ "$(pgrep -cf '[m]easure_all30.sh cnn' 2>/dev/null || echo 0)" -eq 0 ] && [ "$waited" -gt 2 ]; then
            echo "[$(date '+%H:%M:%S')] CNN 測定プロセスが見当たらない。完了マーカーも無いので中断する"
            echo "  -> 手動で 'bash measure_all30.sh cnn' を再実行して埋めること (完了分はスキップされる)"
            exit 1
        fi
    done
fi
echo "[$(date '+%m/%d %H:%M:%S')] ### 段階 1/3: CNN 30 回測定 完了 ($(ls "$E"/results_v30/*.json 2>/dev/null | wc -l)/56) ###"

# ---- 2. DINOv2-L 全段 FP32 (図 6 の対照) ----
echo "[$(date '+%m/%d %H:%M:%S')] ### 段階 2/3: DINOv2-L 全段 FP32 ###"
bash "$E/build_v2_fullfp32.sh"
echo "[$(date '+%m/%d %H:%M:%S')]   -> rc=$?"

# ---- 3. ViT-L 126 パートの 30 回測定 ----
echo "[$(date '+%m/%d %H:%M:%S')] ### 段階 3/3: ViT-L 30 回測定 ###"
bash "$E/measure_all30.sh" vitl
echo "[$(date '+%m/%d %H:%M:%S')]   -> rc=$?"

touch "$E/logs/orin_all30.done"
echo "[$(date '+%m/%d %H:%M:%S')] ##### 全工程 完了 #####"
