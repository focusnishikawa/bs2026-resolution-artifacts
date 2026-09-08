#!/usr/bin/env bash
# 査読指摘 A-3 (group split) の集計を 1 コマンドで通す.
#
#   HPC で集計 -> summary_group.json を Mac へ回収 -> 通常分割との差を表示
#
# ⚠️ 生データ (results/G_cond{A,B}_s<seed>/) は HPC にしかないので、集計は **HPC 側**で行う。
#    Mac に持ってくるのは集計結果の JSON 1 個だけである。
# ⚠️ A1/A5/A-4 (配備精度) とは系統が違うので refresh_a1.sh とは分けてある。
#    こちらは**別の分割で学習し直した**結果であり、本編の数値を置き換えるものではない。
#    レビューの言う「探索的結果 (本編) / 確認的結果 (group split)」の後者にあたる。
#
# 前提となるジョブ (いずれも hpciaiss1):
#   train/run_group_split_boundary.sh   本線の学習・評価          (30 シード)
#   train/fill_group_eval.sh            UCX で落ちた評価の穴埋め
#   train/fill_group_condA_dinov2.sh    条件 A の DINOv2-L を可能にする追加学習
#
# usage:
#   bash train/refresh_a3.sh              # 集計して回収
#   SKIP_COLLECT=1 bash train/refresh_a3.sh   # 回収済みの JSON で表示だけ
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R" || exit 1
H=ufsi0002@hpciaiss1
P=/work/gfsi/ufsi0002/bs2026-resolution

echo "##### A-3 (group split) 集計 $(date '+%F %T') #####"

if [ "${SKIP_COLLECT:-0}" != "1" ]; then
    echo
    echo "----- 1. HPC 側で集計 -----"
    # ⚠️ ssh の失敗を握り潰さない (rc=255 の Broken pipe を「データ無し」と誤判定した前例あり)。
    #    ControlMaster が詰まることがあるので多重化を使わない。
    ok=0
    for t in 1 2 3; do
        if ssh -n -o ControlPath=none -o ConnectTimeout=25 "$H" \
                "cd $P && python3 train/collect_group_split.py"; then
            ok=1; break
        fi
        echo "   ⚠️ HPC での集計に失敗した (試行 $t/3)"
        sleep 5
    done
    [ "$ok" = "1" ] || { echo "[中止] HPC へ 3 回とも繋がらなかった"; exit 1; }

    echo
    echo "----- 2. 集計結果の回収 -----"
    scp -q "$H:$P/results/summary_group.json" results/ \
        || { echo "[中止] summary_group.json を回収できなかった"; exit 1; }
    echo "   results/summary_group.json を回収した"
else
    echo
    echo "----- 1-2. 集計・回収はスキップ (SKIP_COLLECT=1) -----"
fi

echo
echo "----- 3. 中身の確認 -----"
[ -f results/summary_group.json ] || { echo "[中止] results/summary_group.json が無い"; exit 1; }
python3 - <<'PY'
import json
j = json.load(open("results/summary_group.json"))
seeds = j["seeds"]
print("  seed 数     : %d 本 %s" % (len(seeds), seeds if len(seeds) <= 32 else "..."))
print("  境界構成    : %s"
      % ", ".join("%s x %d" % (m, len(v)) for m, v in j["boundary"].items()))
short = []
for m, rs in j["models"].items():
    for r, e in rs.items():
        for mode in ("A", "B"):
            n = e[mode]["n_seed"]
            if n < len(seeds):
                short.append("cond%s %s r%s (%d/%d)" % (mode, m, r, n, len(seeds)))
if short:
    print("  ⚠️ シードが欠けている組 %d 件:" % len(short))
    for s in short[:12]:
        print("     " + s)
    if len(short) > 12:
        print("     ... 他 %d 件" % (len(short) - 12))
else:
    print("  ✅ 全構成・両条件でシードがそろっている")
c = j.get("compare")
if c:
    print("  比較対象    : %s (共通 %d seed)" % (c["source"], len(c["common_seeds"])))
PY

echo
echo "##### 完了 $(date '+%F %T') #####"
echo "  集計結果: results/summary_group.json"
echo "  ⚠️ これは**確認的結果**であり本編の数値を置き換えるものではない。"
echo "     ipsj8b の限界節に「選択が入れ替わるか」として書く"
