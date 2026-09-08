#!/usr/bin/env bash
# fgpu0 (Orin Nano) の **ViT-L 配備精度 CSV** を Mac へ回収する (査読指摘 A-4).
#
# 回収するもの:
#   preds_vitl_30seed/s<NN>/{dinov2_l,dinov3_l}_r<N>.csv   test 1,882 枚の全数推論
#   logs/vitl_acc_chain.log                                測定ログ (失敗・定数出力の確認用)
#
# ⚠️ ラベルは既存の results/preds_30seed/labels.npy を使う。ViT-L も CNN と**同じ
#    test セット**を同じ順序で流している (入力 bin は prep_inputs_orin.py が同じ
#    rel_paths 順で作る) ので、labels を取り直す必要は無い。
#
# ⚠️ seed は **43-71 の 29 本**である。CNN 4 種は 30 本 (42-71) だが、ViT-L の seed 42 は
#    onnx_seeds/ でなく旧 onnx/ 由来で N=32 だけ別経路 (アドホックな static 系) なので
#    混ぜない。集計側 (collect_deploy_acc.py) が n_seed を必ず表示する。
#
# ⚠️ 測定中でも実行してよい。書きかけの CSV も持ってくるが、集計側が行数
#    (1,882 + ヘッダ) で弾くので害は無い。
#
# 転送量は 29 seed x 28 構成そろって約 90 MB。数 GB 未満なので rsync でよい
# (数 GB 以上は Archaea tools + hpfp.j-focus.jp を使う規約。ここは対象外)。
#
# usage: bash train/fetch_vitl_preds.sh
set -u
H=ufsi0002@fgpu0
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R" || exit 1

mkdir -p results/preds_vitl_30seed logs

echo "===== ViT-L 配備精度 CSV の回収 $(date '+%F %T') ====="
# ⚠️ macOS の rsync は openrsync (rsync 2.6.9 互換) なので --info= 系は使えない。
#    ⚠️ ssh の ControlMaster が詰まると Broken pipe (rc=255) になる。失敗を握り潰さない。
if rsync -a "$H:$E/preds_vitl_30seed/" results/preds_vitl_30seed/; then
    echo "  rsync OK"
else
    echo "  ⚠️ rsync 失敗 (rc=$?)。ssh が通っているか確かめること"
fi

scp -q "$H:$E/logs/vitl_acc_chain.log" logs/ 2>/dev/null \
    || echo "  ⚠️ チェーンログを取れなかった"

echo
echo "===== 回収結果 ====="
tot=0
for d in results/preds_vitl_30seed/s*/; do
    [ -d "$d" ] || continue
    n=$(ls "$d"*.csv 2>/dev/null | wc -l | tr -d ' ')
    tot=$((tot+n))
    printf "  %-8s CSV %2s / 28\n" "$(basename "$d")" "$n"
done
ns=$(ls -d results/preds_vitl_30seed/s* 2>/dev/null | wc -l | tr -d ' ')
echo "  合計 CSV ${tot} 件 / seed ${ns} ディレクトリ"
echo "  (そろったときの期待値: CSV 812 件 / seed 29 ディレクトリ = seed 43-71 x 28 構成)"

if [ -f results/preds_30seed/labels.npy ]; then
    echo "  ラベル: results/preds_30seed/labels.npy を流用する (CNN と同じ test セット)"
else
    echo "  ⚠️ results/preds_30seed/labels.npy が無い。先に train/fetch_30seed_preds.sh を実行すること"
fi
