#!/usr/bin/env bash
# 30 シード配備精度 (test+val) を 2 段で回す。
#
# ⭐ 順序に意味がある。
#   第 1 段 seed 53-71: まだ test を測っていないシード。**ビルド 1 本で test と val を両方**測る。
#   第 2 段 seed 43-52: すでに test を測り終えたシード。**val だけ**測る (エンジンは作り直しになる)。
#
#   この順序だと **test が出そろう時刻は v1 のまま (9/4 午後)** で、A1 の再計算を
#   val の完了を待たずに暫定で始められる。逆順 (43 から) にすると test の完走まで
#   val の 17 時間が先に挟まり、test 完走が 9/5 へずれる。総所要は同じである。
#
# 起動: nohup setsid bash run_30seed_chain_v2.sh > logs/deploy_acc_30seed_v2_master.log 2>&1 < /dev/null &
# 完了: logs/deploy_acc_30seed_v2_all.done  (第 1 段だけの完了は logs/deploy_acc_30seed_v2.done)
set -u
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
rm -f logs/deploy_acc_30seed_v2_all.done

echo "########## 第 1 段: seed 53-71 (test+val 同時) $(date '+%F %T') ##########"
bash run_30seed_deploy_acc_v2.sh 53 71
rc1=$?
echo "########## 第 1 段 終了 rc=$rc1 $(date '+%F %T') ##########"

echo
echo "########## 第 2 段: seed 43-52 (val の追加) $(date '+%F %T') ##########"
bash run_30seed_deploy_acc_v2.sh 43 52
rc2=$?
echo "########## 第 2 段 終了 rc=$rc2 $(date '+%F %T') ##########"

echo
echo "===== チェーン完了 $(date '+%F %T') ====="
echo "test CSV: $(find preds_30seed     -name '*.csv' | wc -l) 件 (期待 1624 = 29 seed x 56)"
echo "val  CSV: $(find preds_30seed_val -name '*.csv' | wc -l) 件 (期待 1624)"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ] && touch logs/deploy_acc_30seed_v2_all.done
exit 0
