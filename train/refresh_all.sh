#!/usr/bin/env bash
# 実験データが増えたときに、集計 -> 作図 -> 論文ビルドまでを一発で通す.
#
# 30 seed 化と 30 回測定は数日かけて少しずつ揃うので、そのたびに手で 5 本のコマンドを
# 正しい順序・正しいマシンで叩くのは間違えやすい (実際に Mac で HPC 用スクリプトを
# 実行して summary_v2.json を nan で壊す事故を起こした)。順序と実行場所をここに固定する。
#
# 実行場所は **Mac**。HPC 側の集計だけ ssh 越しに走らせ、結果を持ってくる。
#   1. [HPC] collect_v2.py       評価 JSON -> summary_v2.json   (完全な seed だけを自動検出)
#   2. [Mac] summary_v2.json と Orin の測定結果を回収
#   3. [Mac] collect_vitl_edge.py 相当は HPC 側にあるので ssh で実行し JSON を回収
#   4. [Mac] analyze_all.py      -> final_tables.json
#   5. [Mac] optimal_n.py        -> optimal_n.json (論文の中核表)
#   6. [Mac] make_figs.py        -> figs/ と figs_en/
#   7. [Mac] 論文を再ビルドしてページ数・エラー数を表示
#
# usage: bash train/refresh_all.sh [--skip-fetch]
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
HPC=hpciaiss1
WORK=/work/gfsi/ufsi0002/bs2026-resolution
EDGE=/home1/gfsi/ufsi0002/bs2026-resolution-edge
# ⚠️ HPC のシステム python には numpy が無い。numpy を使う集計は必ず SIF 経由で走らせる
#    (collect_v2.py は標準ライブラリだけなのでシステム python で動く)
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP="apptainer exec --env PYTHONNOUSERSITE=1 --bind /work --bind /home1 $SIF"
SKIP_FETCH=0
[ "${1:-}" = "--skip-fetch" ] && SKIP_FETCH=1

cd "$HERE" || exit 1
echo "===== 1. HPC で評価を集計 (完全な seed のみ) ====="
ssh -o ConnectTimeout=30 "$HPC" "cd $WORK && python3 train/collect_v2.py --root ." | tail -4
rc=$?
if [ $rc -ne 0 ]; then
    echo "[NG] HPC 側の集計が失敗した (rc=$rc)。評価 JSON が揃っているか確認すること"
    exit 1
fi

if [ $SKIP_FETCH -eq 0 ]; then
    echo ""
    echo "===== 2. 集計結果と Orin の測定を Mac へ回収 ====="
    scp -q -o ConnectTimeout=30 "$HPC:$WORK/results/summary_v2.json" results/ && echo "  summary_v2.json"
    # Orin の 30 回測定 (CNN) は /home1 が NFS 共有なので HPC 経由で取れる
    mkdir -p results/orin/results_v30
    scp -q -o ConnectTimeout=30 "$HPC:$EDGE/results_v30/*.json" results/orin/results_v30/ 2>/dev/null \
        && echo "  results_v30/ $(ls results/orin/results_v30/*.json 2>/dev/null | wc -l) 件"

    echo ""
    echo "===== 3. ViT-L のエッジ実測を集計して回収 ====="
    ssh -o ConnectTimeout=30 "$HPC" "cd $WORK && $APP python3 train/collect_vitl_edge.py" | tail -8
    scp -q -o ConnectTimeout=30 "$HPC:$WORK/results/orin_vitl/vitl_edge_summary.json" \
        results/orin_vitl/ 2>/dev/null && echo "  vitl_edge_summary.json"
fi

echo ""
echo "===== 4. 集計表 (final_tables.json) ====="
python3 train/analyze_all.py 2>&1 | grep -E "^\[orin\]|30 回測定|表 [0-9]|saved" | head -12

echo ""
echo "===== 5. 中核表 (optimal_n.json) ====="
python3 train/optimal_n.py 2>&1 | grep -E "構成数|表 A|表 D|saved|^ *0\.9[0-9]" | head -12

echo ""
echo "===== 6. 作図 (日本語 + 英語) ====="
python3 train/make_figs.py > /dev/null 2>&1 && echo "  figs/ 更新"
FIG_LANG=en python3 train/make_figs.py > /dev/null 2>&1 && echo "  figs_en/ 更新"
cp figs/*.png paper/ipsj/figs/ && cp figs_en/*.png paper/preprint/figs/ && echo "  論文側へ反映"

echo ""
echo "===== 7. 論文の再ビルド ====="
echo "--- IPSJ 和文 ---"
bash paper/ipsj/build.sh 2>&1 | tail -5
echo "--- 英語プレプリント ---"
bash paper/preprint/build.sh 2>&1 | tail -4

echo ""
echo "===== 完了。**tex 中の数値は自動では書き換わらない** ====="
echo "  図と JSON は最新になったが、本文・表にベタ書きした数値は手で直す必要がある。"
echo "  差し替え対象: 表2 (Orin latency)、表3 (ViT-L)、表 acctarget、abstract、はじめに、おわりに"
echo "  最新値は results/optimal_n.json と results/final_tables.json を参照すること。"
