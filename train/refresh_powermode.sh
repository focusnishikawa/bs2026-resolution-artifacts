#!/usr/bin/env bash
# 15W / MAXN_SUPER の測定を回収 -> 集計 -> 作図まで一発で通す.
#
# 測定は fgpu0 (Jetson Orin Nano) で走らせるが、出力先 /home1 は hpciaiss1 と NFS 共有
# なので回収は hpciaiss1 経由で取れる (fgpu0 に測定中の余計な負荷をかけない)。
#
#   1. [Mac] results_trt10_{15w,maxn}/*.json を回収
#   2. [Mac] collect_powermode.py -> results/orin/powermode_trt10.json
#   3. [Mac] fig8_powermode を日英で生成し、論文側の figs/ へ反映
#   4. [Mac] 本文へ貼る数値を表示
#
# 測定が途中でも安全に何度でも流せる (対になった構成が無ければ図は作らない)。
#
# usage: bash train/refresh_powermode.sh [--no-fetch]
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
HPC=hpciaiss1
EDGE=/home1/gfsi/ufsi0002/bs2026-resolution-edge
FETCH=1
[ "${1:-}" = "--no-fetch" ] && FETCH=0

cd "$HERE" || exit 1
mkdir -p results/orin/results_trt10_15w results/orin/results_trt10_maxn \
         results/orin/results_trt10_vitl_15w results/orin/results_trt10_vitl_maxn \
         results/orin/results_prep_trt10

if [ "$FETCH" -eq 1 ]; then
    echo "===== 1. 測定結果を回収 ====="
    # CNN 56 構成 (1 構成 = 1 エンジン) と ViT-L (1 構成 = 複数パートの連鎖) の 4 系統
    for pair in "CNN  15w:results_trt10_15w" "CNN  maxn:results_trt10_maxn" \
                "ViTL 15w:results_trt10_vitl_15w" "ViTL maxn:results_trt10_vitl_maxn"; do
        name="${pair%%:*}"; dir="${pair##*:}"
        # ⚠️ まだ 1 件も無いモードでは scp が失敗するので、失敗しても続行する
        scp -q -o ConnectTimeout=30 "$HPC:$EDGE/$dir/*.json" "results/orin/$dir/" 2>/dev/null
        echo "  $name: $(ls "results/orin/$dir"/*.json 2>/dev/null | wc -l | tr -d ' ') 件"
    done
    # ⭐ アイドル電力。これが無いと net_energy_mJ (推論による増分) が出ない
    scp -q -o ConnectTimeout=30 "$HPC:$EDGE/results_trt10_idle.json" \
        "results/orin/" 2>/dev/null
    if [ -f results/orin/results_trt10_idle.json ]; then
        echo "  idle: あり (net_energy_mJ を算出できる)"
    else
        echo "  idle: まだ無い (idle_power.sh 未完了。net_energy_mJ は出ない)"
    fi
    # ⭐ 前処理を両電力モードで測り直したもの (measure_prep_mode.sh の出力)。
    #    これが無いと optimal_n.py --src trt10-* は「新環境 GPU + 旧環境 CPU 前処理」の
    #    混合を避けるために中止する (混ぜないことが目的なのでフォールバックさせない)
    scp -q -o ConnectTimeout=30 "$HPC:$EDGE/results_prep_trt10/*.json" \
        "results/orin/results_prep_trt10/" 2>/dev/null
    np=$(ls results/orin/results_prep_trt10/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  prep: ${np}/6 件 (A/B/deploy x 15w/maxn)$([ "$np" -eq 0 ] && echo ' … measure_prep_mode.sh 未実行')"
fi

echo ""
echo "===== 2. 集計 ====="
python3 train/collect_powermode.py || exit 1

echo ""
echo "===== 3. 作図 (fig8 のみ。他の図には触らない) ====="
python3 -c "import sys; sys.path.insert(0,'train'); import make_figs; make_figs.fig8_powermode()"
FIG_LANG=en python3 -c "import sys; sys.path.insert(0,'train'); import make_figs; make_figs.fig8_powermode()"
if [ -f figs/fig8_powermode.png ]; then
    cp figs/fig8_powermode.png paper/ipsj/figs/
    cp figs_en/fig8_powermode.png paper/preprint/figs/
    echo "  論文側へ反映 (⚠️ 和文は extractbb が要るので build.sh が xbb を作り直す)"
fi

echo ""
echo "===== 4. 本文へ貼る数値 ====="
python3 - <<'PY'
import json, os
# PM_ROOT は collect_powermode.py と同じ検証用の差し替え口 (本番では指定しない)
p = os.path.join(os.environ.get("PM_ROOT") or "results/orin", "powermode_trt10.json")
d = json.load(open(p))
cov = d["coverage"]
if not all(c["complete"] for c in cov.values()):
    print("  ⚠️ まだ 56 構成そろっていない。この数値は暫定である")
    for m, c in cov.items():
        print("     %-11s %2d/%d" % (m, c["n"], c["of"]))
def r(x):
    return "%.2f--%.2f (中央値 %.2f)" % (x["min"], x["max"], x["median"])

idle = d.get("idle_mW")
print("  アイドル電力    : %s"
      % (" / ".join("%s %s mW" % (k, v) for k, v in idle.items()) if idle
         else "未測定 (net_energy_mJ は出ない)"))
print("")
print("--- CNN 56 構成 ---")
ov = d.get("overall") or {}
if not ov:
    print("  対になった構成がまだ無い")
else:
    print("  全体 n=%d" % ov["n_paired"])
    print("  MAXN/15W 高速化 : %s 倍" % r(ov["speedup_maxn_over_15w"]))
    if ov.get("power_ratio"):
        print("  電力比 (VDD_IN) : %s 倍" % r(ov["power_ratio"]))
    if ov.get("energy_ratio"):
        print("  エネルギー比    : %s 倍" % r(ov["energy_ratio"]))
    print("")
    print("  モデル別 (高速化 / エネルギー比):")
    for m, v in d["by_model"].items():
        s, e = v["speedup_maxn_over_15w"], v.get("energy_ratio")
        print("    %-10s %s  |  %s" % (m, r(s), r(e) if e else "-"))
    print("")
    print("  N=112 の代表値:")
    print("    %-10s %8s %8s %8s %8s %8s" % ("model", "15W ms", "MAXN ms", "倍率", "15W mJ", "MAXN mJ"))
    for m in ["mnv4", "effb0", "resnet50", "vit_small"]:
        c = d["by_config"].get("%s_r112" % m)
        if not c or "15W" not in c or "MAXN_SUPER" not in c:
            continue
        a, b = c["15W"], c["MAXN_SUPER"]
        print("    %-10s %8.3f %8.3f %8.2f %8.2f %8.2f" % (
            m, a["median_ms"], b["median_ms"], c["speedup_maxn_over_15w"],
            a.get("energy_mJ", float("nan")), b.get("energy_mJ", float("nan"))))

# ⭐ ViT-L は分割チェーンなので latency は段の総和・エネルギーは Sum(P_i x t_i)
vl = d.get("vitl")
print("")
print("--- ViT-L 分割チェーン ---")
if not vl:
    print("  測定 JSON がまだ無い")
else:
    for mode, c in vl["coverage"].items():
        print("  %-11s 配備 %3d/%d パート (%s)"
              % (mode, c["n_deploy_parts"], c["of_deploy"],
                 "完了" if c["complete"] else "測定中"))
    if vl["incomplete_chains"]:
        print("  ⚠️ 段が欠けている連鎖 %d 件 (比較から除外): %s"
              % (len(vl["incomplete_chains"]), vl["incomplete_chains"][:3]))
    vo = vl.get("overall_deploy") or {}
    if not vo:
        print("  対になった連鎖がまだ無い")
    else:
        print("  配備 n=%d" % vo["n_paired"])
        print("  MAXN/15W 高速化 : %s 倍" % r(vo["speedup_maxn_over_15w"]))
        if vo.get("energy_ratio"):
            print("  エネルギー比    : %s 倍" % r(vo["energy_ratio"]))
        print("")
        print("  N=112 の代表値 (連鎖合計):")
        print("    %-12s %8s %8s %8s %8s %8s"
              % ("model", "15W ms", "MAXN ms", "倍率", "15W mJ", "MAXN mJ"))
        for m in ["dinov2_l", "dinov3_l"]:
            c = vl["by_chain"].get("%s_r112" % m)
            if not c or "speedup_maxn_over_15w" not in c:
                continue
            a, b = c["15W"], c["MAXN_SUPER"]
            print("    %-12s %8.3f %8.3f %8.2f %8.2f %8.2f" % (
                m, a["latency_ms"], b["latency_ms"], c["speedup_maxn_over_15w"],
                a.get("energy_mJ", float("nan")), b.get("energy_mJ", float("nan"))))
PY

echo ""
echo "===== 完了。**tex 中の数値は自動では書き換わらない** ====="
echo "  results/orin/powermode_trt10.json を見て本文・表を手で直すこと。"
