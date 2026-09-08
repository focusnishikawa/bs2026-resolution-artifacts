#!/usr/bin/env bash
# fgpu0 (Orin Nano) の 30 シード配備精度 CSV を Mac へ回収する.
#
# 回収するもの:
#   preds_30seed/s<NN>/*.csv       test 1,882 枚の全数推論   -> results/preds_30seed/
#   preds_30seed_val/s<NN>/*.csv   val  1,876 枚の全数推論   -> results/preds_30seed_val/
#   inputs/labels.npy              test の正解ラベル
#   inputs_val/labels.npy          val  の正解ラベル
#   logs/deploy_acc_30seed_v2_master.log   測定ログ (失敗・定数出力の確認用)
#
# 転送量は全 1,624 x 2 件そろって約 360 MB。**数 GB 未満なので rsync でよい**
# (数 GB 以上は Archaea tools + hpfp.j-focus.jp を使う規約。ここは対象外)。
#
# ⚠️ 測定中でも実行してよい。rsync は書きかけの CSV も持ってくるが、集計側
#    (collect_deploy_acc.py) が行数で弾くので害はない。
#
# usage: bash train/fetch_30seed_preds.sh
set -u
H=ufsi0002@fgpu0
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R" || exit 1

mkdir -p results/preds_30seed results/preds_30seed_val logs

echo "===== 回収 $(date '+%F %T') ====="
for pair in "preds_30seed:results/preds_30seed:inputs" \
            "preds_30seed_val:results/preds_30seed_val:inputs_val"; do
    src="${pair%%:*}"; rest="${pair#*:}"; dst="${rest%%:*}"; indir="${rest#*:}"
    echo "--- $src -> $dst"
    # ⚠️ macOS の rsync は **openrsync (protocol 29 / rsync 2.6.9 互換)** で、
    #    rsync 3.x の --info=stats2 は無い。オプションは 2.6.9 の範囲に収めること。
    #    中身は s<NN>/*.csv だけなのでフィルタも使わずディレクトリごと持ってくる。
    rsync -a "$H:$E/$src/" "$dst/" || echo "  ⚠️ rsync 失敗 ($src)"
    scp -q "$H:$E/$indir/labels.npy" "$dst/labels.npy" 2>/dev/null \
        || echo "  ⚠️ labels.npy を取れなかった ($indir)"
done

scp -q "$H:$E/logs/deploy_acc_30seed_v2_master.log" logs/ 2>/dev/null \
    || echo "  ⚠️ マスターログを取れなかった"

echo
echo "===== 回収結果 ====="
for d in results/preds_30seed results/preds_30seed_val; do
    n=$(find "$d" -name '*.csv' | wc -l | tr -d ' ')
    s=$(ls -d "$d"/s* 2>/dev/null | wc -l | tr -d ' ')
    printf "%-28s CSV %5s 件 / seed %2s ディレクトリ  labels=%s\n" \
           "$d" "$n" "$s" "$([ -f "$d/labels.npy" ] && echo あり || echo なし)"
done
echo "(そろったときの期待値: CSV 1680 件 / seed 30 ディレクトリ = seed 42-71 x 56 構成)"
echo "  ⚠️ seed 42 は他の 29 本と別経路で測る。ONNX が onnx_seeds/ に無く \`onnx/\` にあるため"
echo "     fgpu0 で onnx_seeds/s42 -> ../onnx を張り、run_seed42_chain.sh が第 2 段の後に測る"
