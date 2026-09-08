#!/usr/bin/env bash
# ViT-L 全 14 水準のエッジ実測を 1 本で流すラッパー (ビルド -> 全テスト推論 -> 3 回測定).
#
# ネットワークが切れても走り続けるように、呼び出し側は nohup setsid で起動する。
# 各段は既存成果物をスキップするので、途中で落ちても同じコマンドで再開できる。
#
# usage: nohup setsid bash run_vitl_all.sh <res...> > logs/run_vitl_all_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
RES="$*"
[ -z "$RES" ] && RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
rm -f "$E/logs/run_vitl_all.done"

echo "[$(date '+%m/%d %H:%M:%S')] ##### 開始 (水準: $RES) #####"

echo "[$(date '+%m/%d %H:%M:%S')] ### 段階 1/3: エンジンビルド ###"
bash "$E/build_vitl_all.sh" $RES

echo "[$(date '+%m/%d %H:%M:%S')] ### 段階 2/3: 全テスト 1,882 枚の推論 ###"
bash "$E/infer_vitl_all.sh" $RES

echo "[$(date '+%m/%d %H:%M:%S')] ### 段階 3/3: 3 回測定 (--loadEngine) ###"
bash "$E/measure_vitl_all.sh" $RES

touch "$E/logs/run_vitl_all.done"
echo "[$(date '+%m/%d %H:%M:%S')] ##### 全段階完了 #####"
