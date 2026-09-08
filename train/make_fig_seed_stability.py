#!/usr/bin/env python3
"""シード数 k と符号の安定性の図 (査読指摘 B-A2).

`train/seed_stability.py` が書き出した `results/seed_stability.json` を読み、
横軸をシード数 $k$、縦軸を「30 シード平均の符号と一致する割合」とする図を作る。

⭐ **効果量で層別する**のがこの図の要点である。全体の中央値だけを見ると
「$k$ を増やせば安定する」としか読めないが、実際には
**|30 シード平均差| >= 1 ポイントの組は $k=1$ でも 100% 一致**しており、
不安定なのは**効果量が 1 ポイント未満の組だけ**である。

出力: figs/fig_seed_stability.png (FIG_LANG=en なら figs_en/)

usage:
  python3 train/make_fig_seed_stability.py
  FIG_LANG=en python3 train/make_fig_seed_stability.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
EN = os.environ.get("FIG_LANG", "") == "en"
F = os.path.join(HERE, "figs_en" if EN else "figs")
os.makedirs(F, exist_ok=True)

if EN:
    plt.rcParams["font.family"] = "DejaVu Sans"
else:
    for fam in ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "AppleGothic", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e5e5e0", "grid.linewidth": 0.8,
    # ⚠️ 1 段幅へ縮小して刷るので、既定より 1 段大きめにする (査読指摘: 軸・凡例が小さい)
    "font.size": 13, "legend.frameon": False,
    "axes.titlesize": 13.5, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# 検証済み categorical スロット
C_ALL, C_BIG, C_SMALL = "#52514e", "#1baf7a", "#eb6834"


def T(ja, en):
    return en if EN else ja


def main():
    d = json.load(open(os.path.join(R, "seed_stability.json")))
    ks = sorted(int(k) for k in d["by_k"])
    big_pt = d["big_effect_pt"]
    n_big, n_small = d["n_big"], d["n_small"]

    med = [100 * d["by_k"][str(k)]["agree_median"] for k in ks]
    q1 = [100 * d["by_k"][str(k)]["agree_q1"] for k in ks]
    q3 = [100 * d["by_k"][str(k)]["agree_q3"] for k in ks]
    mbig = [100 * d["by_k"][str(k)]["agree_median_big"] for k in ks]
    msml = [100 * d["by_k"][str(k)]["agree_median_small"] for k in ks]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.fill_between(ks, q1, q3, color=C_ALL, alpha=0.13, lw=0,
                    label=T("全78組の四分位範囲", "IQR over all 78 pairs"))
    ax.plot(ks, med, color=C_ALL, lw=2.0, marker="o", ms=6, zorder=4,
            label=T("全78組の中央値", "median, all 78 pairs"))
    ax.plot(ks, mbig, color=C_BIG, lw=2.0, marker="^", ms=7, zorder=5,
            label=T("効果量 $\\geq$ %.0f pt の %d 組" % (big_pt, n_big),
                    "%d pairs with effect $\\geq$ %.0f pt" % (n_big, big_pt)))
    ax.plot(ks, msml, color=C_SMALL, lw=2.0, marker="s", ms=6, zorder=5,
            label=T("効果量 $<$ %.0f pt の %d 組" % (big_pt, n_small),
                    "%d pairs with effect $<$ %.0f pt" % (n_small, big_pt)))
    ax.axhline(50, color="#b9b9b4", lw=1.0, ls=":", zorder=1)
    # ⚠️ 凡例は右下に置くので、この注釈は**左端**へ出す (右端だと凡例と重なる)
    ax.annotate(T("符号が当てずっぽうと同じ水準", "chance level"), (ks[0], 50),
                xytext=(4, 4), textcoords="offset points", ha="left", va="bottom",
                fontsize=11, color="#8a8a85")

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(45, 103)
    ax.set_xlabel(T("使用したシード数 $k$（対数軸）", "number of seeds $k$ used (log scale)"))
    ax.set_ylabel(T("30シード平均の符号と\n一致した部分集合の割合 [%]",
                    "subsets agreeing in sign with\nthe 30-seed mean [%]"))
    ax.set_title(T("効果量が %.0f ポイント以上なら $k=1$ でも符号は変わらない" % big_pt,
                   "With an effect of $\\geq$ %.0f point the sign is stable even at $k=1$" % big_pt))
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(F, "fig_seed_stability.png"))
    print("[fig] %s/fig_seed_stability.png" % os.path.basename(F))
    for k in ks:
        print("   k=%-3d 中央値 %5.1f%%  (効果量大 %5.1f%% / 小 %5.1f%%)"
              % (k, 100 * d["by_k"][str(k)]["agree_median"],
                 100 * d["by_k"][str(k)]["agree_median_big"],
                 100 * d["by_k"][str(k)]["agree_median_small"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
