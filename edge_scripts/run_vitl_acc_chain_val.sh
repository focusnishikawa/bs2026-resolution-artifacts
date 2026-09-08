#!/usr/bin/env bash
# ViT-L 配備精度 (validation) のエンジン構築+全数推論チェーン (fgpu0 側).
#
# 2026-09-08 の再レビュー指摘 A-5 への対応。HPC 側チェーンが立てる export.done を
# 待ってから 1 シードずつ処理する。test のとき 1 シード実測 98 分・29 シードで約 47--53 時間。
#
# ⛔⛔ **1 巡で終わらせない。** 2026-09-07 (S16) に同種の評価が約 50% 落ちる事象を踏んだ。
#      そろうまで最大 MAXROUND 巡する。再実行が安全な理由 = 完了済みシードはマーカーで
#      飛ばし、途中まで書けた CSV は行数チェックで書き直すため壊れた出力が残らない。
#
# usage: nohup setsid bash run_vitl_acc_chain_val.sh > logs/vitl_acc_chain_val.log 2>&1 &
set -u
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
SEEDS=$(seq 43 71)
MAXROUND="${MAXROUND:-6}"
T0=$(date +%s)
echo "===== ViT-L 配備精度 (val) チェーン 開始 $(date '+%F %T') ====="

for round in $(seq 1 "$MAXROUND"); do
    done_n=$(ls logs/vitl_acc_val_s*.done 2>/dev/null | wc -l)
    echo "########## 巡回 ${round}/${MAXROUND} 開始 $(date '+%F %T') (完了 ${done_n}/29) ##########"
    [ "$done_n" -ge 29 ] && break
    for s in $SEEDS; do
        [ -f "logs/vitl_acc_val_s${s}.done" ] && continue
        # 書き出しを待つ (最大 8 時間)。HPC 側が 34 分/シード
        w=0
        while [ ! -f "vitl_seeds/s${s}/export.done" ]; do
            [ "$w" -ge 480 ] && { echo "[abort] seed $s の書き出しを 8 時間待ったが来ない"; exit 1; }
            [ $((w % 30)) -eq 0 ] && echo "[wait] seed $s の ONNX 待ち ($w 分)"
            sleep 60; w=$((w+1))
        done
        echo "########## seed $s (val) 開始 $(date '+%F %T') ##########"
        bash run_vitl_acc_seed_val.sh "$s"
        echo "########## seed $s (val) 終了 rc=$? $(date '+%F %T') ##########"
    done
done

done_n=$(ls logs/vitl_acc_val_s*.done 2>/dev/null | wc -l)
echo "===== 終了 $(date '+%F %T') 完了 ${done_n}/29 ・所要 $(( ($(date +%s)-T0)/3600 )) 時間 ====="
[ "$done_n" -ge 29 ] && touch logs/vitl_acc_chain_val.done
exit 0
