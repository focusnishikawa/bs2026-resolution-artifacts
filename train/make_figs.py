#!/usr/bin/env python3
"""Phase 5: 論文用の図を生成する.

配色は検証済みパレットの categorical スロット順をそのまま使う。
  1 blue #2a78d6 / 2 orange #eb6834 / 3 aqua #1baf7a / 4 yellow #eda100
  5 magenta #e87ba4 / 6 green #008300
折れ線・棒 (隣接ペアのみ比較される形) は 6 スロットまで安全。
散布図は全ペアが同時に見えるため色は 3 スロットまでが安全域なので、
**マーカー形状と直接ラベルを併用**して色だけに依存しない図にする。

出力: figs/*.png (300 dpi)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
F = os.path.join(HERE, "figs")
os.makedirs(F, exist_ok=True)
# 論文用に PDF も併せて出す (dvipdfmx は PNG だと xbb 依存で事故りやすい)
PAPER_PDF = os.environ.get("PAPER_PDF", "") == "1"
# 英語プレプリント用にラベルを差し替える (FIG_LANG=en)。図の中身は変えない
EN = os.environ.get("FIG_LANG", "") == "en"
if EN:
    F = os.path.join(HERE, "figs_en")
    os.makedirs(F, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"


def T(ja, en):
    """日本語ラベルと英語ラベルを切り替える"""
    return en if EN else ja

def _save(fig, name):
    fig.savefig(os.path.join(F, name + ".png"))
    if PAPER_PDF:
        fig.savefig(os.path.join(F, name + ".pdf"))

# 日本語ラベル用 (macOS)
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
    "font.size": 12, "legend.frameon": False,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    # ⭐ PDF は TrueType 埋め込み (42) にする。既定の Type 3 (3) は
    #    **グリフ名を PDF の Name オブジェクトへ ascii encode する**ため、
    #    和文フォント (Hiragino Sans) のグリフ名で UnicodeEncodeError になり
    #    PAPER_PDF=1 が fig1 で必ず落ちていた。42 はグリフ番号で参照するので起きない
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PALETTE = {  # 検証済み categorical スロット (light)
    "mnv4": "#2a78d6", "effb0": "#eb6834", "resnet50": "#1baf7a",
    "vit_small": "#eda100", "dinov2_l": "#e87ba4", "dinov3_l": "#008300",
}
MARKERS = {"mnv4": "o", "effb0": "s", "resnet50": "^", "vit_small": "D",
           "dinov2_l": "v", "dinov3_l": "P"}
LABEL = {"mnv4": "MobileNetV4", "effb0": "EfficientNet-B0", "resnet50": "ResNet50",
         "vit_small": "ViT-S/16", "dinov2_l": "DINOv2-L", "dinov3_l": "DINOv3-L"}
MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]

acc = json.load(open(os.path.join(R, "summary_v2.json")))
# ⭐ データ源の差し替え口 (collect_powermode.py の PM_ROOT と同じ流儀)。既定は従来どおり。
#    Orin 値を TRT 10.3 へ差し替えるときは、analyze_all.py --src trt10-maxn の出力を渡す:
#      FT_JSON=results/final_tables_trt10_maxn.json python3 train/make_figs.py
tables = json.load(open(os.environ.get("FT_JSON") or os.path.join(R, "final_tables.json")))
v6 = json.load(open(os.path.join(R, "v6", "v6_res14_summary.json")))

# ⭐ 査読指摘 A1 の差し替え口。**図 2 (パレート) の精度だけ**を Orin の配備 FP16 へ差し替える:
#      ACC_JSON=results/summary_deploy.json python3 train/make_figs.py
#
# ⚠️ **図 1 には効かせない。** 図 1 は条件 A と条件 B を並べて学習条件の効果を見る図であり、
#    条件 A の配備精度は測っていない (Orin へ載せたのは条件 B のエンジンだけ)。片方だけ
#    差し替えると「サーバの条件 A 対 配備の条件 B」という別物の比較になってしまう。
#    図 1 はサーバ FP32 の実験結果としてそのまま正しいので、キャプションで出典を明示する。
#
# ⚠️ ViT-L (dinov2_l / dinov3_l) は 30 シードの配備精度を測っていないので従来値のままになる
#    (seed 42 実測の argmax 一致率 99.95-100% で半精度の影響がほぼ無いことは確認済み)。
_acc_dep_path = os.environ.get("ACC_JSON")
acc_dep = json.load(open(_acc_dep_path)) if _acc_dep_path else None


def acc_b(m, r):
    """図 2 用の条件 B 精度。配備精度があればそれ、無いモデルは summary_v2.json の値."""
    if acc_dep is not None:
        d = acc_dep["models"].get(m, {}).get(str(r), {}).get("B")
        if d:
            return d["mean"]
    return acc["models"][m][str(r)]["B"]["mean"]


if acc_dep is not None:
    _fb = sorted({m for m in MODELS
                  if not acc_dep["models"].get(m, {}).get("224", {}).get("B")})
    print("[A1] 図 2 の精度を %s (Orin 配備 FP16) へ差し替えた" % _acc_dep_path)
    if _fb:
        print("     ⚠️ %s は配備精度が無いのでサーバ FP32 のまま" % ", ".join(_fb))


def fig1_accuracy_curves():
    """精度 vs 解像度。条件 A / B を並べる (折れ線なので 6 スロット使用可)。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cond, title in [(axes[0], "A", T("条件A: 224学習モデルへの劣化入力", "Regime A: degraded input to a 224-trained model")),
                            (axes[1], "B", T("条件B: 解像度別ネイティブ学習", "Regime B: resolution-native retraining"))]:
        for m in MODELS:
            ys = [acc["models"][m][str(r)][cond]["mean"] for r in RES]
            es = [acc["models"][m][str(r)][cond]["std"] for r in RES]
            ax.plot(RES, ys, color=PALETTE[m], lw=2, marker=MARKERS[m], ms=4,
                    label=LABEL[m], zorder=3)
            ax.fill_between(RES, [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=PALETTE[m], alpha=0.13, lw=0, zorder=2)
        ax.set_xlabel(T("入力解像度 N (px)", "input resolution N (px)"))
        ax.set_title(title, fontsize=12.5)
        ax.set_xticks([16, 64, 112, 160, 224])
        ax.set_ylim(0.15, 1.0)
    axes[0].set_ylabel(T("test accuracy (30 seed 平均)", "test accuracy (mean of 30 seeds)"))
    axes[1].legend(loc="lower right", fontsize=11, ncol=2)
    fig.suptitle(T("解像度と種判別精度 (猛禽6種, n=1882)。帯は 30 seed の標準偏差", "Resolution vs. species accuracy (6 raptor species, n=1882). Bands: sd over 30 seeds"), fontsize=12.5, y=1.02)
    _save(fig, "fig1_accuracy_vs_resolution")
    plt.close(fig)
    print("  fig1_accuracy_vs_resolution.png")


def fig2_pareto():
    """精度-レイテンシのパレート。

    散布図は全ペアが同時に見えるので色 3 スロットが安全域。ここでは 4 系列あるため
    **色 + マーカー形状 + 凡例**の三重で識別できるようにし、色だけに依存しない。
    系列名は凡例に任せ、図中の直接ラベルは注記 1 点だけに絞る (点との重なりを避ける)。
    """
    orin = {m: v["latency"] for m, v in tables["orin_latency"].items()}
    fig, ax = plt.subplots(figsize=(7.8, 5))
    order = ["mnv4", "effb0", "resnet50", "vit_small"]
    # ⭐⭐ 査読指摘 A1: **パレート印と「効率の頂点」は、実際に描いた精度から作り直す。**
    #   ACC_JSON を渡すと縦軸は配備 FP16 精度になるのに、front / best を final_tables
    #   (サーバ FP32) から取ると**図が自分の縦軸と矛盾する**。2026-09-05 に実際に起きた:
    #   配備では front から外れる **ViT-S/16 N=160-224 の 5 点が「パレート最適」の大きい印**
    #   のまま残り、逆に front 上にある **ResNet50 N=208/224 が小さい印**のままだった。
    #   ⚠️ ACC_JSON 無しのときは **tables の値をそのまま使う** (従来の図とバイト一致させるため)。
    # ⛔⛔ 査読指摘 A-4 (2026-09-08): **図と表でパレート判定を同じ集合から取る**。
    #   従来は ①front を小型 4 モデルだけで計算し ②ViT-L は全点を無条件に大きい印で
    #   描いていた。そのため「大きい印＝パレート最適」というキャプションが成り立たず、
    #   例えば DINOv3-L $N$=16 (約 16 ms・0.66) は ResNet50 $N$=32 に支配されるのに
    #   大きい印だった。
    #   ⭐ 直し方 = **optimal_n.py が出した `pareto_gpu_ms` (全 84 候補・validation で選抜)
    #      をそのまま読む**。こうすると「図の大きい印の集合」と「表のパレート集合」は
    #      構成上つねに一致する (別々に計算しない)。
    #   ⚠️ 縦軸は test 精度なので、**val で選ばれた点が test 上では支配されて見えることがある**。
    #      キャプションで「validation で選抜」と明示すること。
    pareto_json = os.environ.get("PARETO_JSON")
    front_pts = None
    if pareto_json and os.path.exists(pareto_json):
        _pj = json.load(open(pareto_json))
        if "pareto_gpu_ms" in _pj:
            front_pts = [{"model": p["model"], "res": p["res"], "acc": acc_b(p["model"], p["res"]),
                          "latency_ms": p["gpu_ms"]} for p in _pj["pareto_gpu_ms"]]
            print("[A4] パレート印は %s の pareto_gpu_ms (%d 点・validation 選抜) を使う"
                  % (os.path.basename(pareto_json), len(front_pts)))
    if front_pts is None:
        if acc_dep is not None:
            # 落とし先: 描いた精度から全 84 候補で計算する (ViT-L も含める)
            _pts = [{"model": m, "res": r, "acc": acc_b(m, r),
                     "latency_ms": orin[m].get(str(r), orin[m].get(r))}
                    for m in order if m in orin for r in RES]
            for _m, _sw in (("dinov2_l", tables.get("orin_vitl", {})
                             .get("resolution_sweep_dinov2_l", {}).get("by_res", {})),
                            ("dinov3_l", tables.get("orin_vitl", {})
                             .get("resolution_sweep_dinov3_l", {}).get("by_res", {}))):
                for _r, _v in _sw.items():
                    _pts.append({"model": _m, "res": int(_r), "acc": acc_b(_m, int(_r)),
                                 "latency_ms": _v["latency_ms"]})
            front_pts = [p for p in _pts
                         if not any(q["latency_ms"] <= p["latency_ms"] and q["acc"] >= p["acc"]
                                    and (q["latency_ms"], q["acc"]) != (p["latency_ms"], p["acc"])
                                    for q in _pts)]
        else:
            front_pts = tables["pareto_front"]
    front = {(p["model"], p["res"]) for p in front_pts}
    for m in order:
        if m not in orin:
            continue
        xs = [orin[m].get(str(r), orin[m].get(r)) for r in RES]
        ys = [acc_b(m, r) for r in RES]
        ax.plot(xs, ys, color=PALETTE[m], lw=1.2, alpha=0.5, zorder=2)
        for r, x, y in zip(RES, xs, ys):
            on = (m, r) in front
            ax.scatter(x, y, s=62 if on else 26, color=PALETTE[m], marker=MARKERS[m],
                       edgecolor="#fcfcfb", linewidth=1.4 if on else 0.6,
                       zorder=5 if on else 3,
                       label=LABEL[m] if r == RES[0] else None)

    # ViT-L (8/19 追加)。分割チェーンで実機動作した。DINOv2-L は 4 水準を実測したので線で結ぶ。
    # 精度は他の点と揃えて条件 B の 30 seed 平均を使う (Orin 実測の acc は本文 3.6)
    vitl = tables.get("orin_vitl", {}).get("models", {})
    sweep = tables.get("orin_vitl", {}).get("resolution_sweep_dinov2_l", {}).get("by_res", {})
    if sweep:
        rs = sorted(int(k) for k in sweep)
        xs = [sweep[str(r)]["latency_ms"] for r in rs]
        ys = [acc_b("dinov2_l", r) for r in rs]
        ax.plot(xs, ys, color=PALETTE["dinov2_l"], lw=1.6, alpha=0.7, zorder=5)
        # ⭐ 査読指摘 A-4: ViT-L も**フロント上の点だけ**を大きい印にする (従来は全点が大)
        sz = [90 if ("dinov2_l", r) in front else 34 for r in rs]
        ax.scatter(xs, ys, s=sz, color=PALETTE["dinov2_l"], marker=MARKERS["dinov2_l"],
                   edgecolor="#fcfcfb", linewidth=1.4, zorder=6,
                   label=T("DINOv2-L (N=16〜224)", "DINOv2-L (N=16-224)"))
    sweep3 = tables.get("orin_vitl", {}).get("resolution_sweep_dinov3_l", {}).get("by_res", {})
    if sweep3:
        rs = sorted(int(k) for k in sweep3)
        xs = [sweep3[str(r)]["latency_ms"] for r in rs]
        ys = [acc_b("dinov3_l", r) for r in rs]
        ax.plot(xs, ys, color=PALETTE["dinov3_l"], lw=1.6, alpha=0.7, zorder=5)
        sz = [90 if ("dinov3_l", r) in front else 34 for r in rs]
        ax.scatter(xs, ys, s=sz, color=PALETTE["dinov3_l"], marker=MARKERS["dinov3_l"],
                   edgecolor="#fcfcfb", linewidth=1.4, zorder=6,
                   label=T("DINOv3-L (N=16〜224, 全段FP32)", "DINOv3-L (N=16-224, all-FP32)"))
    for m, rec in vitl.items():
        if m == "dinov2_l" and sweep:
            continue
        if m == "dinov3_l" and sweep3:
            continue
        x = rec["latency_ms"]
        y = acc_b(m, 32)
        ax.scatter(x, y, s=90 if (m, 32) in front else 34, color=PALETTE[m], marker=MARKERS[m],
                   edgecolor="#fcfcfb", linewidth=1.4, zorder=6,
                   label=T("%s (N=32, 4分割)", "%s (N=32, 4-way split)") % LABEL[m])
    # 注記は 2 点だけ: 効率の頂点と、最高精度だが最も遅い点
    best = max(front_pts, key=lambda p: p["acc"] / p["latency_ms"])
    # 注記は点群と重ならない位置に固定する (軸座標で置き、矢印だけを点へ伸ばす)
    ax.annotate(T("効率の頂点  %s N=%d\n%.3f / %.2f ms", "efficiency peak  %s N=%d\n%.3f / %.2f ms") % (LABEL[best["model"]], best["res"],
                                                        best["acc"], best["latency_ms"]),
                (best["latency_ms"], best["acc"]), textcoords="axes fraction",
                xytext=(0.02, 0.94), ha="left", va="top", fontsize=11, color="#0b0b0b",
                arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=1))
    # 注記の上下関係: 「小型モデルで最高精度」を上、「ViT-L は動く」を下に置く。
    # ⭐ 2 つの注記は**先頭 (左端) を縦に揃える**。前者は左下の小型モデル最高精度点を、
    #    後者は右上の DINOv2-L @32 を指すので、右端で揃えると矢印が X 字に交差する。
    #    左端を揃えると前者の矢印は左端側から左下へ、後者は右端側から右上へ出るので交差しない。
    #    ⚠️ 後者は ha="right" で幅が可変 (和文と英文で長さが違う) なため、
    #    実際に描いてから左端を測る。0.30 は ViT-L 注記が無いときのフォールバック。
    #
    # ⭐⭐ **「どの小型モデルが最高精度か」を決め打ちしない。** 従来は ViT-S/16 @224 を
    #    指して「MobileNetV4 の 5 倍遅い」と固定文字列で書いていたが、どちらも古かった:
    #    ①配備 FP16 (査読指摘 A1) では ViT-S/16 @224 は 0.8801 まで落ち、ResNet50 0.9213 /
    #      EfficientNet-B0 0.9208 を下回って **4 モデル中最下位**になる (2026-09-05 に発見)
    #    ②「5 倍」は TRT 8.5.2 時代の比。新環境ではサーバ FP32 で数えても 2.4 倍にすぎない
    #    → **最高精度の点も倍率も毎回データから決める**。倍率は効率の頂点 (best) との比。
    _small = max(({"model": m, "res": r, "acc": acc_b(m, r),
                   "latency_ms": orin[m].get(str(r), orin[m].get(r))}
                  for m in order if m in orin for r in RES),
                 key=lambda p: (p["acc"], -p["latency_ms"]))
    vx, vy = _small["latency_ms"], _small["acc"]
    _slow = _small["latency_ms"] / best["latency_ms"]
    small_x = 0.30
    if "dinov2_l" in vitl:
        dx = vitl["dinov2_l"]["latency_ms"]
        dy = acc_b("dinov2_l", 32)
        # ⚠️ **レイテンシを固定文字列で書かない。** 「66〜91 ms」は TRT 8.5.2 時代の値で、
        #    S10 の新環境 (TRT 10.3 / MAXN_SUPER) 差し替えのときに取り残されていた
        #    (2026-09-05 に発見)。新環境の実測は DINOv2-L 8.7-27.0 ms / DINOv3-L 16.2-44.0 ms。
        #    最大 N での 2 モデルの実測値から毎回作る。
        _hi = []
        for _sw in (sweep, sweep3):
            if _sw:
                _rmax = max(int(k) for k in _sw)
                _hi.append(_sw[str(_rmax)]["latency_ms"])
        _rng = (T("%.0f〜%.0f ms", "%.0f-%.0f ms") % (min(_hi), max(_hi))) if _hi else \
               (T("%.0f ms", "%.0f ms") % vitl["dinov2_l"]["latency_ms"])
        an = ax.annotate(T("ViT-L は動く (分割)。N を上げると\n精度は最高だが %s かかる",
                           "ViT-L does run (split). Raising N\ngives top accuracy at %s") % _rng,
                         (dx, dy), textcoords="axes fraction", xytext=(0.62, 0.10),
                         ha="right", va="center", fontsize=11, color="#0b0b0b",
                         arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=1))
        fig.canvas.draw()
        small_x = an.get_window_extent().transformed(ax.transAxes.inverted()).x0
    # ⚠️ 高さ 0.40 では 2 行目の文末に ViT-L 側の矢印が刺さる (矢印は右上へ抜けるので、
    #    注記が低い位置にあるほどテキストの右端を横切る)。**2 行分ぶん上げて 0.51 にする**
    #    (この図では 1 行 = axes fraction で約 0.054)。上げても点群には触れない
    #    (x=0.29-0.52 は 1.6-6 ms の帯で、その高さに実測点は無い)。
    #    ⚠️ 英文は "but 5x slower than MobileNetV4" が和文より 1 割ほど長く、0.51 では
    #    まだ "MobileNetV4" の文字を矢印が貫く (拡大して実測)。英文だけ 0.65 まで上げ、
    #    **矢印の終点 (DINOv2-L @32 = axes y 0.554) より上**へ逃がす。
    #    この高さでも 2.5-7 ms の帯に実測点は無いので点群とは重ならない。
    ax.annotate(T("小型モデルで最高精度だが\n%s の %.1f 倍遅い",
                  "best of the small models,\nbut %.1fx slower than %s")
                % ((_slow, LABEL[best["model"]]) if EN
                   else (LABEL[best["model"]], _slow)), (vx, vy),
                textcoords="axes fraction", xytext=(small_x, T(0.51, 0.65)), ha="left", va="center",
                fontsize=11, color="#0b0b0b",
                arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=1))
    ax.set_xscale("log")
    ax.margins(x=0.10, y=0.12)
    ax.set_xlabel(T("Orin Nano レイテンシ (ms/枚、対数軸)", "Orin Nano latency (ms/image, log scale)"))
    ax.set_ylabel(T("test accuracy (条件B, 30 seed 平均)", "test accuracy (regime B, mean of 30 seeds)"))
    # ⭐ 査読指摘 A-4: 大きい印は **validation で選んだ**パレート集合である。
    #   縦軸は test 精度なので、**test 上では支配されて見える点が含まれうる**。断らずに
    #   「Pareto-optimal」と書くと、図が主張する集合を誤って表示することになる。
    ax.set_title(T("精度とレイテンシ。大きい印はvalidationで選んだパレート集合。ViT-L は分割で実機動作 (N=16〜224)",
                   "Accuracy vs. latency. Large markers: validation-selected Pareto set; ViT-L runs when split (N=16-224)"),
                 fontsize=12.5)
    # 凡例は軸の外 (下) へ出す。図中に注記を 3 つ置くため、内側に凡例を置くと必ずどこかで重なる
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, handletextpad=0.4, columnspacing=1.6)
    _save(fig, "fig2_pareto_orin")
    plt.close(fig)
    print("  fig2_pareto_orin.png")


def fig3_pixel_requirement():
    """所要画素数の比較 (棒)。条件 A と B を並べる。"""
    pr = tables["pixel_requirement"]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    x = range(len(MODELS))
    w = 0.38
    a = [pr[m]["A"] for m in MODELS]
    b = [pr[m]["B"] for m in MODELS]
    ax.bar([i - w / 2 for i in x], a, w - 0.02, label=T("条件A (劣化耐性)", "regime A (degradation robustness)"),
           color="#86b6ef", edgecolor="#fcfcfb", linewidth=1)
    ax.bar([i + w / 2 for i in x], b, w - 0.02, label=T("条件B (解像度別学習)", "regime B (resolution-native)"),
           color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va + 2, str(va), ha="center", fontsize=11, color="#52514e")
        ax.text(i + w / 2, vb + 2, str(vb), ha="center", fontsize=11,
                color="#0b0b0b", fontweight="bold")
    ax.set_ylim(0, max(max(a), max(b)) * 1.16)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=18, ha="right")
    ax.set_ylabel(T("必要入力解像度 N (一辺の画素数)", "required input resolution N (pixels per side)"))
    ax.set_title(T("モデル別の必要入力解像度 (条件A と 条件B の比較)", "Required input resolution per model: retraining helps CNNs, hurts ViTs"), fontsize=12.5)
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    _save(fig, "fig3_pixel_requirement")
    plt.close(fig)
    print("  fig3_pixel_requirement.png")


def fig4_fp16():
    """配備エンジンとサーバ FP32 の argmax 一致率.

    CNN と ViT-S/16 は純 FP16、DINOv2-L は混合精度 (p3s1 のみ FP32)、DINOv3-L は全段 FP32。
    ViT-L は代表 6 水準を系列で描く (8/20)。**x 軸はモデルへの実入力**なので、DINOv2-L は
    patch14 の丸めで 14/28/70/112/126/224 に落ち、指定 N (16/32/64/112/128/224) とはずれる。
    DINOv2-L の全段 FP32 系列は、混合精度との差が精度差なのか半精度化の影響なのかを
    切り分けるための対照であり、データが揃っていれば破線で重ねる。
    """
    rows = tables["fp16"]
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["res"]] = r["argmax_agreement"] * 100
    # ⭐ 先に全系列を集めてから描く。y 軸を割るかどうかを全データを見てから決めるため
    series = []   # (x, y, plot kwargs)
    ys = []       # 描画した全系列の値。y 軸範囲をデータから決めるために集める
    xs = [16, 112, 224]
    for m in ["mnv4", "effb0", "resnet50", "vit_small"]:
        vals = [by[m][r] for r in xs]
        series.append((xs, vals, dict(color=PALETTE[m], lw=2,
                                      marker=MARKERS[m], ms=7, label=LABEL[m])))
        ys += vals

    # ViT-L: 指定 N の 6 水準。実入力でプロットする
    LV = [16, 32, 64, 112, 128, 224]
    ov = tables.get("orin_vitl", {})
    for m, key, note, ls, col in [
            ("dinov2_l", "resolution_sweep_dinov2_l", T("混合精度", "mixed precision"), "-", None),
            ("dinov3_l", "resolution_sweep_dinov3_l", T("全段FP32", "all-FP32"), "-", None),
            ("dinov2_l", "resolution_sweep_dinov2_l_fullfp32", T("全段FP32", "all-FP32"), "--", "#8a3d5f")]:
        sw = ov.get(key, {}).get("by_res", {})
        if not sw:
            continue
        px, py = [], []
        for n in LV:
            rec = sw.get(str(n))
            if not rec or rec.get("argmax_agreement") is None:
                continue
            px.append(rec.get("input_res", n))
            py.append(rec["argmax_agreement"] * 100)
        if not px:
            continue
        series.append((px, py, dict(color=col or PALETTE[m], lw=2, ls=ls,
                                    marker=MARKERS[m], ms=7,
                                    mfc="none" if ls == "--" else None,
                                    label="%s (%s)" % (LABEL[m], note))))
        ys += py

    # ⭐ 値は二群に分かれる (CNN と ViT-L の密集帯 98--100% と、ViT-S/16 の低い側)。
    #    1 枚の軸に収めると密集帯が潰れて 5 系列の差が読めないので y 軸を割る。
    #    ⚠️ 分割点は**データの最大ギャップ**から決める。群が分かれなくなれば
    #    ギャップが縮んで自動的に 1 枚の軸へ戻る (しきい値 3.0 ポイント)
    def _lim(vals, frac=0.18, floor=0.15):
        a, b = min(vals), max(vals)
        pad = max(floor, (b - a) * frac)
        return a - pad, b + pad

    sv = sorted(set(round(v, 4) for v in ys))
    gap, gi = max(((sv[i + 1] - sv[i], i) for i in range(len(sv) - 1)), default=(0.0, -1))
    if gap >= 3.0:
        # ⚠️ 凡例を軸の外へ出すぶん縦が伸びる。**figsize で相殺しないと縦横比が変わり、
        #    英文プレプリント (width=0.9\textwidth) が 1 ページ増える**
        fig, (ax_hi, ax_lo) = plt.subplots(
            2, 1, sharex=True, figsize=(7.4, 3.9),
            gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.10})
        for x, y, kw in series:
            ax_hi.plot(x, y, **kw)
            ax_lo.plot(x, y, **dict(kw, label="_nolegend_"))
        a, b = _lim(sv[gi + 1:])
        ax_hi.set_ylim(a, min(100.35, b))     # 一致率は 100% を超えない
        ax_lo.set_ylim(*_lim(sv[:gi + 1]))
        # 軸を割ったことを示す (境界の枠線を消して破断記号を置く)
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(axis="x", bottom=False)
        brk = dict(marker=[(-1, -0.55), (1, 0.55)], ms=9, ls="none",
                   color="0.35", mec="0.35", mew=1.2, clip_on=False)
        ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **brk)
        ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **brk)
        ax_top, ax_bottom = ax_hi, ax_lo
        fig.supylabel(T("サーバ FP32 との argmax 一致率 (%)",
                        "argmax agreement with server FP32 (%)"), fontsize=11)
        # 凡例は軸の外 (下) へ出す。密集帯にも ViT-S/16 の低い側にも被らせない
        h, l = ax_hi.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.015),
                   ncol=3, fontsize=9, frameon=False)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for x, y, kw in series:
            ax.plot(x, y, **kw)
        lo, hi = min(ys), max(ys)
        pad = max(0.6, (hi - lo) * 0.10)
        ax.set_ylim(lo - pad - (hi - lo) * 0.22, min(100.6, hi + pad))
        ax.set_ylabel(T("サーバ FP32 との argmax 一致率 (%)",
                        "argmax agreement with server FP32 (%)"))
        ax.legend(fontsize=9, loc="lower left", ncol=2)
        ax_top = ax_bottom = ax

    ax_bottom.set_xticks([16, 64, 112, 160, 224])
    ax_bottom.set_xlabel(T("入力解像度 N (px)", "input resolution N (px)"))
    ax_top.set_title(T("配備エンジンとサーバ FP32 の argmax 一致率", "Argmax agreement between the deployed Orin engines and server FP32"),
                     fontsize=12.5)
    _save(fig, "fig4_fp16_agreement")
    plt.close(fig)
    print("  fig4_fp16_agreement.png")


def fig5_v6_alert():
    """v6 カスケードの保護対象種アラート。precision と recall は対で読む。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    sysmap = [("condA_flat", "#86b6ef", "o", T("条件A 分離版", "regime A, separate")),
              ("condA_s34", "#2a78d6", "s", T("条件A 統合版", "regime A, merged")),
              ("condB_fix", "#eb6834", "^", T("条件B ResNet50", "regime B, ResNet50")),
              ("condB_dinov2_l", "#e87ba4", "v", T("条件B DINOv2-L", "regime B, DINOv2-L")),
              ("condB_dinov3_l", "#008300", "P", T("条件B DINOv3-L", "regime B, DINOv3-L"))]
    for ax, met, title in [(axes[0], "alert_precision", T("適合率 (precision)", "precision")),
                           (axes[1], "alert_recall", T("再現率 (recall)", "recall"))]:
        for key, col, mk, lab in sysmap:
            if key not in v6:
                continue
            pts = [(int(r), v6[key][r][met]) for r in sorted(v6[key], key=int)
                   if v6[key][r].get(met) is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2,
                    marker=mk, ms=5, label=lab)
        ax.set_xlabel(T("入力解像度 N (px)", "input resolution N (px)"))
        ax.set_title(title, fontsize=12.5)
        ax.set_xticks([16, 64, 112, 160, 224])
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(T("値", "value"))
    axes[1].legend(fontsize=11, loc="upper left")
    fig.suptitle(T("v6 IUCN カスケードの保護対象種アラート。低解像度で recall が上がって見えるのは precision 崩壊の裏返し", "Protected-species alert in the IUCN cascade: apparent recall gains at low N are the flip side of precision collapse"), fontsize=12.5, y=1.02)
    _save(fig, "fig5_v6_alert")
    plt.close(fig)
    print("  fig5_v6_alert.png")


def fig6_vitl_edge():
    """ViT-L のエッジ実測 (8/19 追加)。

    左: エッジ/サーバ比。**同じ N=32 で揃えて**比べる (N をまたぐと比が意味を失う)。
    右: 4 分割チェーンの段別内訳。修正前は一部の段が入力に依存しない定数を返しており、
        計算が消えるぶん**速く**出ていた。速度が正常性の証拠にならないことを示す。
    """
    vitl = tables.get("orin_vitl", {}).get("models", {})
    if not vitl:
        print("  [skip] fig6 (orin_vitl なし)")
        return
    h200_r32 = {}
    for m in MODELS:
        p = os.path.join(R, "orin", "results_h200", "%s_r32.json" % m)
        if os.path.exists(p):
            h200_r32[m] = json.load(open(p))["gpu_compute_mean_ms_median_of_reps"]
    orin_r32 = {m: v["latency"]["32"] for m, v in tables["orin_latency"].items()}
    for m, rec in vitl.items():
        orin_r32[m] = rec["latency_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    fig.subplots_adjust(wspace=0.45)

    # --- 左: エッジ/サーバ比 (N=32) ---
    ax = axes[0]
    ms = [m for m in MODELS if m in orin_r32 and m in h200_r32]
    ratio = [orin_r32[m] / h200_r32[m] for m in ms]
    bars = ax.bar(range(len(ms)), ratio, 0.6,
                  color=[PALETTE[m] for m in ms], edgecolor="#fcfcfb", linewidth=1)
    for i, (b, v) in enumerate(zip(bars, ratio)):
        ax.text(i, v + max(ratio) * 0.02, "%.1f" % v, ha="center", fontsize=11,
                color="#0b0b0b", fontweight="bold" if v > 10 else "normal")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([LABEL[m] for m in ms], rotation=18, ha="right")
    ax.set_ylim(0, max(ratio) * 1.16)
    ax.set_ylabel(T("Orin / H200 の latency 比 (倍)", "Orin / H200 latency ratio"))
    ax.set_title(T("エッジ/サーバ比 (N=32)", "Edge/server latency ratio (N=32)"), fontsize=12.5)

    # --- 右: 最終構成の段別内訳 (どの段が時間を食うか) ---
    ax = axes[1]
    shades = ["#86b6ef", "#2a78d6", "#eb6834", "#f0a58a", "#1baf7a"]
    labels, offsets, nmax = [], [], 0
    # ラベルの浮かせ幅と軸上限を同じ基準にするため、合計は先に出しておく
    tmax = max(sum(rec["parts"].values()) for rec in vitl.values())
    for i, (m, rec) in enumerate(vitl.items()):
        bottom = 0.0
        for k, (pname, ms_) in enumerate(rec["parts"].items()):
            ax.bar(i, ms_, 0.55, bottom=bottom, color=shades[k],
                   edgecolor="#fcfcfb", linewidth=1)
            bottom += ms_
        nmax = max(nmax, len(rec["parts"]))
        ax.text(i, bottom + tmax * 0.02, "%.1f" % bottom, ha="center", fontsize=10,
                fontweight="bold", color="#0b0b0b")
        labels.append(LABEL[m])
        offsets.append(i)
    ax.set_xticks(offsets)
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    # ⚠️ 上限はデータから決める。旧環境の値に合わせた 42 ms 固定のままだと、
    #    TRT 10.3 の実測 (最大 16.6 ms) では軸の 6 割が空く
    ax.set_ylim(0, tmax * 1.16)
    ax.set_ylabel(T("GPU compute time の合計 (ms)", "total GPU compute time (ms)"))
    ax.set_title(T("配備構成の段別内訳 (N=32)", "Per-stage breakdown (N=32)"),
                 fontsize=12.5)
    # 凡例は軸の外へ (内側だと y 軸の目盛ラベルと重なる)
    ax.legend(handles=[Line2D([], [], color=c, lw=8, label=T("第%d段", "stage %d") % (k + 1))
                       for k, c in enumerate(shades[:nmax])],
              fontsize=11, ncol=nmax, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              columnspacing=1.2)
    fig.suptitle(T("ViT-L の Orin Nano 実測 (分割チェーン, N=32)", "ViT-L measured on Orin Nano (split chains, N=32)"), fontsize=12.5, y=1.03)
    _save(fig, "fig6_vitl_edge")
    plt.close(fig)
    print("  fig6_vitl_edge.png")


def fig7_latency_scaling():
    """解像度を下げるとどれだけ速くなるかを 6 モデルで揃えて見る (8/20 追加).

    左: 実測 latency (対数)。右: N=16 を 1 とした正規化。
    ViT-L も全 14 水準を同一の準備経路 (生の分割) で測り直したので、6 モデルを同じ土俵で比べられる。
    正規化すると、総画素数の比 196 に対して推論時間の比が 1.48-2.71 にとどまることが 1 枚で分かる。
    """
    orin = {m: v["latency"] for m, v in tables["orin_latency"].items()}
    sw = {"dinov2_l": tables.get("orin_vitl", {}).get("resolution_sweep_dinov2_l", {}).get("by_res", {}),
          "dinov3_l": tables.get("orin_vitl", {}).get("resolution_sweep_dinov3_l", {}).get("by_res", {})}

    def series(m):
        if m in sw and sw[m]:
            return [sw[m][str(r)]["latency_ms"] for r in RES]
        d = orin.get(m, {})
        return [d.get(str(r), d.get(r)) for r in RES]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ends = []   # 右パネルの終端ラベル。重ならないよう後でまとめて置く
    for m in MODELS:
        ys = series(m)
        if any(y is None for y in ys):
            continue
        axes[0].plot(RES, ys, color=PALETTE[m], lw=2, marker=MARKERS[m], ms=4, label=LABEL[m])
        rel = [y / ys[0] for y in ys]
        axes[1].plot(RES, rel, color=PALETTE[m], lw=2, marker=MARKERS[m], ms=4)
        ends.append((rel[-1], m))
    axes[0].set_yscale("log")
    axes[0].set_ylabel(T("Orin Nano の推論時間 (ms、対数)", "Orin Nano latency (ms, log scale)"))
    axes[0].set_title(T("実測レイテンシ", "Measured latency"), fontsize=12.5)
    axes[1].set_ylabel(T("N=16 を 1 とした比", "latency relative to N=16"))
    axes[1].set_title(T("N=16 で正規化 (右端が $t_{224}/t_{16}$)", "Normalised to N=16 (right edge is $t_{224}/t_{16}$)"), fontsize=12.5)
    for ax in axes:
        ax.set_xlabel(T("入力解像度 N (px)", "input resolution N (px)"))
        ax.set_xticks([16, 64, 112, 160, 224])
    axes[1].set_xlim(10, 268)
    # 終端ラベルは値が近いと重なるので、下から順に最小間隔を確保して置き直す
    lo, hi = axes[1].get_ylim()
    gap = (hi - lo) * 0.055
    ends.sort()
    placed = []
    for v, m in ends:
        y = v if not placed else max(v, placed[-1] + gap)
        placed.append(y)
        axes[1].annotate("%.2f" % v, (RES[-1], y), textcoords="offset points",
                         xytext=(8, -4), fontsize=10, color=PALETTE[m], fontweight="bold")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.15, -0.16), ncol=6,
                   fontsize=10.5, handletextpad=0.4, columnspacing=1.2)
    # ⚠️ 比の範囲はベタ書きせず、右パネルに実際に描いた終端値から出す。
    #    旧稿は 1.8〜4.1 のまま残っており、同じ図に描かれる実値 (1.48〜2.71) と食い違っていた
    _r = [v for v, _ in ends]
    fig.suptitle(T("総画素数の比 196 に対し、推論時間の比は %.2f〜%.2f にとどまる" % (min(_r), max(_r)),
                   "Total input pixel count differs by 196x, but latency changes by only %.2f-%.2fx" % (min(_r), max(_r))),
                 fontsize=12.5, y=1.02)
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "fig7_latency_scaling")
    plt.close(fig)
    print("  fig7_latency_scaling.png")


def fig8_powermode():
    """15W と MAXN_SUPER の差 (8/31 追加).

    両モードとも同一環境 (JetPack 6.2 / L4T R36.4.3 / TensorRT 10.3) で測っているので、
    **電力モードの差だけを純粋に見ている**。論文の既存値 (JetPack 5.1.2 / TRT 8.5.2) とは
    TensorRT 版が交絡するため混ぜない。

    左: 推論時間 (実線 MAXN_SUPER / 破線 15W)。右: 1 推論あたりのエネルギー。
    エッジ機器では「速いが電力を食う」ので、時間だけでなくエネルギーで見ないと判断できない。

    ⚠️ 測定が途中でも落ちないようにしてある (対になった構成が無ければ図を作らずに戻る)。
    """
    path = os.path.join(R, "orin", "powermode_trt10.json")
    if not os.path.exists(path):
        print("  fig8_powermode: powermode_trt10.json が無いので飛ばす")
        return
    pm = json.load(open(path))
    cfg = pm.get("by_config", {})
    models = [m for m in ["mnv4", "effb0", "resnet50", "vit_small"]
              if any(c["model"] == m and "15W" in c and "MAXN_SUPER" in c
                     for c in cfg.values())]
    if not models:
        print("  fig8_powermode: 両モードが揃った構成がまだ無いので飛ばす")
        return
    incomplete = not all(c["complete"] for c in pm.get("coverage", {}).values())

    def series(model, mode, key):
        """欠測は None のまま返す (測定中でも描けるように)"""
        xs, ys = [], []
        for r in RES:
            c = cfg.get("%s_r%d" % (model, r))
            if c and mode in c and key in c[mode]:
                xs.append(r)
                ys.append(c[mode][key])
        return xs, ys

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for m in models:
        for mode, ls in (("MAXN_SUPER", "-"), ("15W", "--")):
            xs, ys = series(m, mode, "median_ms")
            axes[0].plot(xs, ys, ls=ls, color=PALETTE[m], lw=2,
                         marker=MARKERS[m], ms=4)
            xs, ys = series(m, mode, "energy_mJ")
            if ys:
                axes[1].plot(xs, ys, ls=ls, color=PALETTE[m], lw=2,
                             marker=MARKERS[m], ms=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(T("推論時間 (ms、対数)", "latency (ms, log scale)"))
    axes[0].set_title(T("推論時間", "Latency"), fontsize=12.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(T("1 推論あたりのエネルギー (mJ、対数)",
                         "energy per inference (mJ, log scale)"))
    axes[1].set_title(T("消費エネルギー (VDD_IN 基準)",
                        "Energy per inference (from VDD_IN)"), fontsize=12.5)
    for ax in axes:
        ax.set_xlabel(T("入力解像度 N (px)", "input resolution N (px)"))
        ax.set_xticks([16, 64, 112, 160, 224])
        # 対数軸の既定ラベル (3x10^0 など) は読みにくいので素の数値にする
        for axis in (ax.yaxis,):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(ScalarFormatter())
            axis.get_major_formatter().set_scientific(False)
            axis.get_minor_formatter().set_scientific(False)
        ax.tick_params(axis="y", which="minor", labelsize=9)

    # 凡例は「モデル = 色/マーカー」と「電力モード = 線種」を分けて示す
    handles = [Line2D([], [], color=PALETTE[m], marker=MARKERS[m], lw=2, ms=5,
                      label=LABEL[m]) for m in models]
    handles += [Line2D([], [], color="#52514e", ls="-", lw=2, label="MAXN_SUPER"),
                Line2D([], [], color="#52514e", ls="--", lw=2, label="15 W")]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(1.15, -0.16),
                   ncol=len(handles), fontsize=10.5, handletextpad=0.4,
                   columnspacing=1.2)

    def rng(d, ja_sep="〜", en_sep="-"):
        """min と max が丸めて同じなら 1 値で書く (「1.45〜1.45 倍」を避ける)"""
        lo, hi = d["min"], d["max"]
        if abs(hi - lo) < 0.005:
            return "%.2f" % lo
        return "%.2f%s%.2f" % (lo, ja_sep if not EN else en_sep, hi)

    ov = pm.get("overall", {}) or {}
    sp, en = ov.get("speedup_maxn_over_15w"), ov.get("energy_ratio")
    if sp and en:
        title = T("MAXN_SUPER は 15 W より %s 倍速いが、"
                  "1 推論あたりのエネルギーは %s 倍になる" % (rng(sp), rng(en)),
                  "MAXN_SUPER is %sx faster than 15 W, "
                  "but uses %sx the energy per inference" % (rng(sp), rng(en)))
    else:
        title = T("電力モードの比較", "Power-mode comparison")
    if incomplete:
        title += T("  【測定中の暫定図】", "  [PROVISIONAL: measurement in progress]")
    fig.suptitle(title, fontsize=12.5, y=1.02)
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "fig8_powermode")
    plt.close(fig)
    print("  fig8_powermode.png%s" % (" (暫定)" if incomplete else ""))


if __name__ == "__main__":
    print("[figs]")
    fig1_accuracy_curves()
    fig2_pareto()
    fig3_pixel_requirement()
    fig4_fp16()
    fig5_v6_alert()
    fig6_vitl_edge()
    fig7_latency_scaling()
    fig8_powermode()
    print("[done] %s" % F)
