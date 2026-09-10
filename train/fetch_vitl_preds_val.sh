#!/usr/bin/env bash
# fgpu0 (Orin Nano) の **ViT-L 配備精度 CSV (validation)** を Mac へ回収する (査読指摘 A-5).
#
# 回収するもの:
#   preds_vitl_30seed_val/s<NN>/{dinov2_l,dinov3_l}_r<N>.csv   val 1,876 枚の全数推論
#   logs/vitl_acc_chain_val.log                                測定ログ (失敗・定数出力の確認用)
#
# ⚠️ ラベルは既存の results/preds_30seed_val/labels.npy を使う。ViT-L も CNN と**同じ
#    val セット**を同じ順序で流している (入力 bin は prep_inputs_vitl_val.sh →
#    prep_inputs_vitl.py が同じ splits/val.csv の順で作る) ので、labels を取り直す必要は無い。
#    ⚠️ val は 1,876 枚で **test の 1,882 枚とは枚数が違う**。test 側の labels.npy を
#    間違って渡さないこと (集計側が行数で弾くが、無用の欠測になる)。
#
# ⚠️ seed は **43-71 の 29 本**である。CNN 4 種は 30 本 (42-71) だが、ViT-L の seed 42 は
#    onnx_seeds/ でなく旧 onnx/ 由来で N=32 だけ別経路 (アドホックな static 系) なので
#    混ぜない。集計側 (collect_deploy_acc.py) が n_seed を必ず表示する。
#
# ⚠️ fgpu0 への直結 ssh は **banner exchange timeout で落ちることがある**。
#    /home1 は NFS 共有なので、そのときは踏み台側から同一実体を回収できる:
#      H=ufsi0002@hpciaiss1 bash train/fetch_vitl_preds_val.sh
#    read-only の回収なので hpciaiss1 で計算を回さないルールには抵触しない。
#
# ⚠️ 測定中でも実行してよい。書きかけの CSV も持ってくるが、集計側が行数
#    (1,876 + ヘッダ) で弾くので害は無い。
#
# 転送量は 29 seed x 28 構成そろって約 90 MB。数 GB 未満なので rsync でよい
# (数 GB 以上は Archaea tools + hpfp.j-focus.jp を使う規約。ここは対象外)。
#
# usage: bash train/fetch_vitl_preds_val.sh
#        H=ufsi0002@hpciaiss1 bash train/fetch_vitl_preds_val.sh   # fgpu0 が落ちているとき
set -u
H="${H:-ufsi0002@fgpu0}"
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R" || exit 1

mkdir -p results/preds_vitl_30seed_val logs

echo "===== ViT-L 配備精度 CSV (val) の回収 $(date '+%F %T')  [$H] ====="
# ⚠️ macOS の rsync は openrsync (rsync 2.6.9 互換) なので --info= 系は使えない。
#    ⚠️ ssh の ControlMaster が詰まると Broken pipe (rc=255) になる。失敗を握り潰さない。
if rsync -a "$H:$E/preds_vitl_30seed_val/" results/preds_vitl_30seed_val/; then
    echo "  rsync OK"
else
    echo "  ⚠️ rsync 失敗 (rc=$?)。ssh が通っているか確かめること"
    echo "     (fgpu0 が落ちていたら H=ufsi0002@hpciaiss1 で同一の /home1 を読める)"
fi

scp -q "$H:$E/logs/vitl_acc_chain_val.log" logs/ 2>/dev/null \
    || echo "  ⚠️ チェーンログを取れなかった"

echo
echo "===== 回収結果 ====="
tot=0
for d in results/preds_vitl_30seed_val/s*/; do
    [ -d "$d" ] || continue
    n=$(ls "$d"*.csv 2>/dev/null | wc -l | tr -d ' ')
    tot=$((tot+n))
    printf "  %-8s CSV %2s / 28\n" "$(basename "$d")" "$n"
done
ns=$(ls -d results/preds_vitl_30seed_val/s* 2>/dev/null | wc -l | tr -d ' ')
echo "  合計 CSV ${tot} 件 / seed ${ns} ディレクトリ"
echo "  (そろったときの期待値: CSV 812 件 / seed 29 ディレクトリ = seed 43-71 x 28 構成)"

if [ -f results/preds_30seed_val/labels.npy ]; then
    echo "  ラベル: results/preds_30seed_val/labels.npy を流用する (CNN と同じ val セット)"
else
    echo "  ⚠️ results/preds_30seed_val/labels.npy が無い。先に train/fetch_30seed_preds.sh を実行すること"
fi
