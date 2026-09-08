#!/usr/bin/env python3
"""「短い推論時間で識別精度が高くなる入力解像度 N はいくつか」に答えるための集計.

本 PJ の主問いをこの 1 本に集約する。精度 (条件 B) と Orin 実測 latency を突き合わせ、
  (1) モデル横断のパレートフロント        どの (モデル, N) が「他に負けない」か
  (2) モデルごとの飽和点                  精度がほぼ頭打ちになる最小の N とそのときの時間
  (3) 時間制約つきの最良精度              「T ms 以内」で選べる最良構成
  (4) 精度目標つきの最短時間              「精度 A 以上」を最短で満たす構成
を出す。(3)(4) は実務での選び方そのものなので、論文の中核表になる。

時間は 3 通りで出す。

  gpu_ms        : trtexec の GPU Compute Time (モデルだけの時間)
  e2e_ms        : **オフライン評価経路**の前処理を含む 1 枚あたりの時間
  e2e_deploy_ms : **配備経路**の前処理を含む 1 個体あたりの時間

⚠️ **この 2 つの経路は別物であり、N を下げる利得が正反対になる** (2026-08-24 に実測で判明)。

  オフライン評価経路: 個体ごとに切り出し済みの JPEG を開く
      decode -> resize(256) -> crop(224) -> N へ縮小 -> 正規化
      デコード 3.4 ms とリサイズ 2.3 ms が解像度に依存しないため前処理が下限を作り、
      N=224->80 で **15% しか減らない**
  配備経路: 4K フレームは検出器が既に 1 回デコード済み。判別器はメモリ上の配列から
      bbox 切り出し -> N へリサイズ -> 正規化 だけを個体ごとに行う (JPEG デコードは発生しない)
      個体あたり 0.20-2.12 ms しかかからず、しかも **N に強く依存する** (正規化が O(N^2))。
      N=224->80 で **47% 減る**

実運用で効くのは配備経路のほうである。オフライン経路の値は、評価スクリプトの実装に
由来する上乗せを含むため、配備時の処理能力を見積もる用途には使えない。

データ源 (いずれも analyze_all.py が更新するので、30 seed / 30 回測定へ差し替わると自動追従する):
  results/summary_v2.json    精度 (条件 A/B の seed 平均・標準偏差)
  results/final_tables.json  Orin latency (CNN/ViT-S) と ViT-L の解像度スイープ
  results/orin/results/prep_timing_B.json       前処理 (オフライン評価経路)
  results/orin/results/prep_timing_deploy.json  前処理 (配備経路)

⭐ データ源 (--src) — 2026-08-31 に追加
  analyze_all.py / collect_vitl_edge.py と同じ流儀。JetPack 6.2 / TRT 10.3 で測り直した
  GPU 時間に切り替えるとき、**前処理も同じ電力モードで測り直したものへ揃える**。

  legacy      JP5.1.2 / TRT 8.5.2 / 15W  final_tables.json + results/prep_timing_{B,deploy}.json
  trt10-15w   JP6.2   / TRT 10.3  / 15W  final_tables_trt10_15w.json  + results_prep_trt10/*_15w.json
  trt10-maxn  JP6.2   / TRT 10.3  / MAXN final_tables_trt10_maxn.json + results_prep_trt10/*_maxn.json

  ⚠️⚠️ **旧環境の前処理へ黙ってフォールバックしない。** 既存の prep_timing_*.json は
     JetPack 5.1.2 / 15W / CPU 1.51 GHz の値なので、新環境の GPU 時間に足すと e2e が
     「新環境 GPU + 旧環境 CPU 前処理」の混合になる (MAXN では CPU が 1.51 -> 1.73 GHz)。
     前処理は mnv4 N=224 で e2e の 40% を占めるため、これは主要な結論の大半に効く。
     見つからないときは**案内を出して止まる**。

usage:
    python3 train/optimal_n.py [--acc-drop 1.0]        # 旧環境 (既定・出力名も従来どおり)
    python3 train/optimal_n.py --src trt10-maxn        # JP6.2 / TRT 10.3 / MAXN_SUPER
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")

# ---- データ源 (main() で --src により設定する。既定は旧環境で振る舞い不変) ----
SRC = "legacy"
# 前処理を両電力モードで測り直したものの置き場 (measure_prep_mode.sh:34 の OUT を回収したもの)
PREP_TRT10_DIR = "results_prep_trt10"
SRC_ENV = {"legacy": {"jetpack": "5.1.2", "l4t": "R35.4.1", "trt": "8.5.2",
                      "power_mode": "15W", "suffix": None},
           "trt10-15w": {"jetpack": "6.2", "l4t": "R36.4.3", "trt": "10.3.0",
                         "power_mode": "15W", "suffix": "15w"},
           "trt10-maxn": {"jetpack": "6.2", "l4t": "R36.4.3", "trt": "10.3.0",
                          "power_mode": "MAXN_SUPER", "suffix": "maxn"}}


def is_trt10():
    return SRC != "legacy"


def tables_path():
    """analyze_all.py の出力先と一致させる"""
    if is_trt10():
        return os.path.join(R, "final_tables_%s.json" % SRC.replace("-", "_"))
    return os.path.join(R, "final_tables.json")


def prep_path(which):
    """前処理 JSON の置き場。which は "B" (オフライン評価経路) か "deploy" (配備経路)"""
    if is_trt10():
        return os.path.join(R, "orin", PREP_TRT10_DIR,
                            "prep_timing_%s_%s.json" % (which, SRC_ENV[SRC]["suffix"]))
    return os.path.join(R, "orin", "results", "prep_timing_%s.json" % which)

RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]
LABEL = {"mnv4": "MobileNetV4", "effb0": "EfficientNet-B0", "resnet50": "ResNet50",
         "vit_small": "ViT-S/16", "dinov2_l": "DINOv2-L", "dinov3_l": "DINOv3-L"}
PATCH = {"dinov2_l": 14, "dinov3_l": 16}
# 実務で意味のある区切り。1 カメラの想定最大スループットは 4K (3840x1920) 30 fps なので、
# **1 カメラでフレームを取りこぼさない上限は 33.3 ms** である。それより速い側は 10 ms のみ残す
# (これ以上切り詰めても前処理が下限を作るため得るものが少ない)。
# 50 / 100 ms は、抽出された個体を 1 回だけ判別する運用 (フレーム周期に縛られない) を想定する。
# ⚠️ 2 / 5 ms や 60 fps (16.7 ms) は本システムの運用条件から外れるので扱わない。
TIME_BUDGETS = [10.0, 33.3, 50.0, 100.0]

# ---- 実システムの構成 ----
# **1 ユニット = 周辺カメラ 3 台 + 追跡用 4K PTZ 1 台 (計 4 台) + Jetson Orin Nano 8GB 1 台**。
# カメラはいずれも 4K (3840x1920)。1 カメラの最大は 30 fps だが、
# **予測・回避システムとしての最低目標は 4 台すべてを 10 fps** で処理すること。
# **360 度監視はこのユニットを 2 組 (カメラ 8 台・Orin 2 台) 並べて達成する**。
# 各ユニットは独立に 4 カメラを担当するので、**1 カメラあたりの処理能力はユニット数に依らない**。
# この構成は v6 論文 (EcoEvoRxiv, DOI: 10.32942/X2KD69, 2026-08-18 公開) と同一である。
NCAM = 4
NUNIT = 2
# 1 個体あたりの判別回数。同一個体と同定できている間は原則 1 回だが、風車から 3 分 / 2 分 /
# 1 分後に回転半径の 2 倍・3 倍の圏内へ到達すると予測された時点で追加するため最大 4 回。
CLASSIFY_PER_BIRD = (1, 4)
ACC_TARGETS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.96]

# 配備経路で基準にする bbox の一辺 [px]。検出器が返す個体の画素サイズにあたる。
# 224 はオフライン経路 (224 基準クロップから N へ縮小) と揃えるための値、
# 112 は遠方個体を想定した対照。
# ⚠️ bbox と N が一致する水準ではリサイズが不要になり、その点だけ下振れする。
#   これは実測どおりの値であって補間はしない (下振れは短縮率を**小さく**見せる保守側の扱い)。
DEPLOY_BBOX = 224
DEPLOY_BBOX_ALT = 112


def input_res(model, n):
    """モデルへ実際に入る一辺の画素数 (patch 倍数へ丸めた値)"""
    p = PATCH.get(model)
    if not p:
        return n
    return n if n % p == 0 else max(p, p * round(n / p))


def load(select_acc_path=None, report_acc_path=None, select_fallback_path=None):
    acc = json.load(open(os.path.join(R, "summary_v2.json")))
    # 査読指摘 A5: 構成選択は validation、報告は test。両者を分けて持つ。
    sel = json.load(open(select_acc_path)) if select_acc_path else None
    # 査読指摘 A1: 報告する精度は**配備するエンジン (Orin FP16) のもの**でなければならない。
    # ⚠️ ViT-L (dinov2_l / dinov3_l) は 30 シードの配備精度を測っていない (1 シード 126 パートの
    #    分割チェーンになるため)。無い構成は summary_v2.json へ落とすが、**黙っては落とさない**。
    #    根拠: seed 42 の実測で ViT-L のサーバとの argmax 一致率は 99.95-100% であり、
    #    半精度化の影響がほぼ無い (DINOv2-L は混合精度・DINOv3-L は全段 FP32 で配備するため)。
    rep = json.load(open(report_acc_path)) if report_acc_path else None
    # ⭐ 選択側の落とし先 (報告側と対称にするためのもの)。
    #    ⚠️ これが無いと **ViT-L が候補から丸ごと消える**。ViT-L の配備精度は測らない設計
    #    (1 シード 126 パート = 30 シードで 3,780 エンジン) なので、測定完了を待っても
    #    val 側には永久に現れない。落とし先はサーバ FP32 の validation (summary_val.json)
    #    で、こちらは 84/84 構成が 30 シードそろっている。
    sel_fb = json.load(open(select_fallback_path)) if select_fallback_path else None
    tpath = tables_path()
    if not os.path.exists(tpath):
        raise SystemExit(
            "[中止] %s が無い。\n"
            "        先に  python3 train/analyze_all.py --src %s  を実行すること。" % (tpath, SRC))
    tables = json.load(open(tpath))

    # ⚠️ 旧環境の前処理へ黙って落ちない。混ぜると e2e が「新環境 GPU + 旧環境 CPU」になる。
    bpath = prep_path("B")
    if not os.path.exists(bpath):
        raise SystemExit(
            "[中止] %s が無い。\n"
            "        %s の前処理をまだ測っていない。fgpu0 で次を実行し、結果を\n"
            "        results/orin/%s/ へ回収すること:\n"
            "          bash measure_prep_mode.sh all\n"
            "        ⚠️ 旧環境 (JetPack 5.1.2 / 15W / CPU 1.51GHz) の prep_timing_B.json で\n"
            "           代用してはいけない。e2e が新旧の混合になる。"
            % (bpath, SRC, PREP_TRT10_DIR))
    prep = json.load(open(bpath))["by_res"]

    dep_path = prep_path("deploy")
    dep = json.load(open(dep_path))["per_bird"] if os.path.exists(dep_path) else None
    if dep is None:
        print("[警告] %s が無いので配備経路の集計を省略する" % dep_path)

    orin = {m: v["latency"] for m, v in tables.get("orin_latency", {}).items()}
    sweep = {
        "dinov2_l": tables.get("orin_vitl", {}).get("resolution_sweep_dinov2_l", {}).get("by_res", {}),
        "dinov3_l": tables.get("orin_vitl", {}).get("resolution_sweep_dinov3_l", {}).get("by_res", {}),
    }

    pts = []
    for m in MODELS:
        for n in RES:
            # 報告用の精度。--report-acc があればそちらを優先し、無い構成だけ従来値へ落とす。
            a, a_src = {}, "server_fp32"
            if rep is not None:
                a = rep["models"].get(m, {}).get(str(n), {}).get("B", {})
                if a:
                    a_src = "deployed_fp16"
            if not a:
                a = acc["models"].get(m, {}).get(str(n), {}).get("B", {})
            if not a:
                continue
            if m in sweep and sweep[m]:
                g = sweep[m].get(str(n), {}).get("latency_ms")
            else:
                d = orin.get(m, {})
                g = d.get(str(n), d.get(n))
            if g is None:
                continue
            pre = prep[str(n)]["total"]["mean"]
            # acc = 報告用 (test)。acc_sel = 選択用 (既定は test と同じ)。
            acc_sel_src = "deployed_fp16"
            if sel is not None:
                sa = sel["models"].get(m, {}).get(str(n), {}).get("B", {})
                if not sa and sel_fb is not None:
                    # 配備の val が無い構成 (ViT-L) はサーバ FP32 の val へ落とす。
                    # ⚠️ 黙って test で選ばない。落ちた対象は必ず表示・記録する
                    sa = sel_fb["models"].get(m, {}).get(str(n), {}).get("B", {})
                    acc_sel_src = "server_fp32"
                if not sa:
                    continue     # 落とし先も無い構成は候補にしない
                acc_sel, acc_sel_sd = sa["mean"], sa.get("std", 0.0)
            else:
                acc_sel, acc_sel_sd = a["mean"], a.get("std", 0.0)
                acc_sel_src = a_src
            rec = {
                "model": m, "res": n, "input_res": input_res(m, n),
                "acc": a["mean"], "acc_sd": a.get("std", 0.0), "acc_source": a_src,
                "acc_sel": acc_sel, "acc_sel_sd": acc_sel_sd,
                "acc_sel_source": acc_sel_src,
                # ⚠️ collect_v2.py が書くキーは "n_seed" (単数)。
                #    "n"/"count" を見ていたため常に None になっていた (2026-08-26 修正)。
                "n_seeds": a.get("n_seed", a.get("n", a.get("count"))),
                "gpu_ms": round(g, 3), "prep_ms": round(pre, 3),
                "e2e_ms": round(pre + g, 3),
            }
            if dep is not None:
                pd = dep[str(DEPLOY_BBOX)][str(n)]["total_cv"]
                pd_alt = dep[str(DEPLOY_BBOX_ALT)][str(n)]["total_cv"]
                rec["prep_deploy_ms"] = round(pd, 3)
                rec["e2e_deploy_ms"] = round(pd + g, 3)
                rec["e2e_deploy_bbox%d_ms" % DEPLOY_BBOX_ALT] = round(pd_alt + g, 3)
            pts.append(rec)
    return pts, acc, (dep is not None)


def pareto(pts, tkey):
    """時間最小・精度最大のパレートフロント。同時間なら高精度、同精度なら短時間が残る"""
    out = []
    for p in pts:
        dominated = any(
            (q[tkey] <= p[tkey] and q["acc_sel"] >= p["acc_sel"]) and
            (q[tkey] < p[tkey] or q["acc_sel"] > p["acc_sel"])
            for q in pts)
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda x: x[tkey])


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-drop", type=float, default=1.0,
                    help="飽和とみなす精度の許容低下 [pt] (既定 1.0)")
    ap.add_argument("--select-acc", default=None,
                    help="構成選択に使う精度 JSON (査読指摘 A5)。"
                         "既定は None = 報告と同じ集合で選ぶ従来動作。"
                         "results/summary_deploy_val.json を渡すと "
                         "**選択は validation の配備精度・報告は test の配備精度** になる")
    ap.add_argument("--report-acc", default=None,
                    help="報告する精度 JSON (査読指摘 A1)。"
                         "既定は None = results/summary_v2.json (サーバ PyTorch FP32)。"
                         "results/summary_deploy.json を渡すと **Orin の配備 FP16 エンジンの精度**で"
                         "表・パレート・結論を作る。配備精度が無いモデル (ViT-L) だけは"
                         "summary_v2.json へ落とし、その旨を表示する")
    ap.add_argument("--select-fallback", default="results/summary_val.json",
                    help="選択側 (validation) の落とし先 JSON。"
                         "--select-acc に無い構成をここから引く。"
                         "⚠️ **ViT-L は配備精度を測らない設計なので、これが無いと候補から"
                         "丸ごと消える** (表 5 の 0.95/0.96 が達成不能になる)。"
                         "none を渡すと落とさず候補から外す従来動作へ戻る")
    ap.add_argument("--src", default="legacy", choices=sorted(SRC_ENV),
                    help="データ源 (既定 legacy = JetPack 5.1.2 / TRT 8.5.2)")
    ap.add_argument("--out", default=None,
                    help="出力先 JSON (既定はデータ源ごとの名前。legacy は従来どおり)")
    args = ap.parse_args()
    SRC = args.src

    # 落とし先は --select-acc があるときだけ効く (無ければ選択=報告なので出番がない)
    sel_fb_path = None
    if args.select_acc and args.select_fallback and args.select_fallback.lower() != "none":
        sel_fb_path = args.select_fallback
        if not os.path.exists(sel_fb_path):
            raise SystemExit(
                "[中止] 選択側の落とし先 %s が無い。\n"
                "        ViT-L は配備の validation 精度を測らない設計なので、これが無いと\n"
                "        表 5 の 0.95/0.96 が「達成不能」になる。\n"
                "        サーバ FP32 の validation 集計 (collect_val.py の出力) を渡すか、\n"
                "        意図的に外すなら --select-fallback none を明示すること。" % sel_fb_path)
    pts, _, has_dep = load(args.select_acc, args.report_acc, sel_fb_path)
    rep_name = os.path.basename(args.report_acc) if args.report_acc else "results/summary_v2.json"
    if args.select_acc:
        print("[A5] 構成選択: %s (validation) / 報告: %s (test)" % (args.select_acc, rep_name))
        sel_fb = sorted({p["model"] for p in pts
                         if p.get("acc_sel_source") == "server_fp32"})
        if sel_fb:
            print("     ⚠️ %s は配備の validation 精度が無いので、選択側は %s "
                  "(サーバ FP32 の validation) で行う"
                  % (", ".join(sel_fb), os.path.basename(sel_fb_path)))
        elif sel_fb_path is None:
            print("     ⚠️ 落とし先なし (--select-fallback none)。"
                  "選択側に無い構成は候補から外れる")
    else:
        print("[A5] ⚠️ 選択と報告が同じ集合 (test)。--select-acc で分離できる")
        sel_fb = []
    fb = sorted({p["model"] for p in pts if p["acc_source"] == "server_fp32"}) \
        if args.report_acc else []
    if args.report_acc:
        print("[A1] 報告精度: %s (**Orin の配備 FP16 エンジンの実測**)" % rep_name)
        if fb:
            print("     ⚠️ %s は配備精度を測っていないのでサーバ FP32 で報告する "
                  "(seed 42 の argmax 一致率 99.95-100%%)" % ", ".join(fb))
    else:
        print("[A1] ⚠️ 報告精度が **サーバ PyTorch FP32** である。配備するのは Orin の FP16 "
              "エンジンで、ViT-S/16 は N>=96 で 3-6 ポイント落ちる。"
              "--report-acc results/summary_deploy.json で差し替えられる")
    tkeys = ["gpu_ms", "e2e_ms"] + (["e2e_deploy_ms"] if has_dep else [])
    tname = {"gpu_ms": "GPU 時間", "e2e_ms": "オフライン評価経路", "e2e_deploy_ms": "配備経路"}
    out = {"note": ("「短い推論時間で識別精度が高くなる N」の集計。"
                    "精度は条件 B (解像度別ネイティブ学習) の seed 平均、"
                    "時間は Orin Nano 実測。e2e_ms はオフライン評価経路の前処理を含み、"
                    "e2e_deploy_ms は配備経路 (4K フレームは検出器が既にデコード済みで、"
                    "判別器は bbox 切り出し -> N へリサイズ -> 正規化 のみ) の前処理を含む。"
                    "**N を下げる利得はこの 2 経路で正反対になる**"),
           "n_points": len(pts), "acc_drop_pt": args.acc_drop,
           "deploy_bbox_px": DEPLOY_BBOX if has_dep else None,
           # 査読指摘 A1/A5 の出典。どの精度で選び、どの精度で報告したかを必ず残す。
           "acc_report_source": rep_name,
           "acc_select_source": (os.path.basename(args.select_acc) if args.select_acc
                                 else rep_name),
           "acc_report_is_deployed": bool(args.report_acc),
           "acc_fallback_models": fb,
           # 選択側の落とし先。どのモデルをサーバ FP32 の val で選んだかを必ず残す
           "acc_select_fallback_source": (os.path.basename(sel_fb_path)
                                          if sel_fb_path else None),
           "acc_select_fallback_models": sel_fb}
    if is_trt10():
        # 旧環境の値と取り違えないよう、出典をファイルへ残す (GPU も前処理も新環境である)
        e = SRC_ENV[SRC]
        out["source"] = SRC
        out["measurement_env"] = {k: e[k] for k in ("jetpack", "l4t", "trt", "power_mode")}
        out["latency_source"] = os.path.basename(tables_path())
        out["prep_source"] = {"offline": os.path.basename(prep_path("B")),
                              "deploy": os.path.basename(prep_path("deploy")) if has_dep else None}
    nseed = {p.get("n_seeds") for p in pts if p.get("n_seeds")}
    out["n_seeds"] = sorted(x for x in nseed if x) or None
    print("[構成数] %d 点 (seed 数: %s)" % (len(pts), out["n_seeds"]))
    if is_trt10():
        e = SRC_ENV[SRC]
        print("[データ源] %s (JetPack %s / L4T %s / TRT %s / %s) — GPU も前処理も新環境"
              % (SRC, e["jetpack"], e["l4t"], e["trt"], e["power_mode"]))

    # ---- (2) モデルごとの飽和点 ----
    print("\n=== 表 A: モデルごとの飽和点 (自身の最高精度から -%.1f pt 以内に入る最小の N) ===" % args.acc_drop)
    print("%-16s %5s %6s %8s %8s %9s %9s %8s" %
          ("モデル", "N*", "実入力", "精度", "最高精度", "GPU[ms]", "e2e[ms]", "対224"))
    sat = {}
    for m in MODELS:
        ps = [p for p in pts if p["model"] == m]
        if not ps:
            continue
        best = max(p["acc"] for p in ps)
        cand = [p for p in ps if p["acc"] >= best - args.acc_drop / 100.0]
        star = min(cand, key=lambda p: p["res"])
        p224 = next((p for p in ps if p["res"] == 224), None)
        sat[m] = {"res": star["res"], "input_res": star["input_res"], "acc": round(star["acc"], 4),
                  "best_acc": round(best, 4), "gpu_ms": star["gpu_ms"], "e2e_ms": star["e2e_ms"],
                  "gpu_speedup_vs_224": round(p224["gpu_ms"] / star["gpu_ms"], 2) if p224 else None,
                  "e2e_speedup_vs_224": round(p224["e2e_ms"] / star["e2e_ms"], 2) if p224 else None}
        # ⚠️ N=224 が欠損していると比は出せない。従来は `or 0` で "0.00x" と出ており、
        #    「速度比ゼロ」と読めてしまった (trt10 の部分データで実際に出た)。
        #    56 構成そろっていれば必ず値があるので legacy の表示は変わらない。
        sp = sat[m]["gpu_speedup_vs_224"]
        print("%-16s %5d %6d %8.4f %8.4f %9.2f %9.2f %8s" %
              (LABEL[m], star["res"], star["input_res"], star["acc"], best,
               star["gpu_ms"], star["e2e_ms"],
               ("%.2fx" % sp) if sp is not None else "(N=224 未測定)"))
    out["saturation_point"] = sat

    # ---- (1) パレートフロント ----
    for tkey in tkeys:
        name = tname[tkey]
        fr = pareto(pts, tkey)
        print("\n=== 表 B(%s): パレートフロント (%d 点) ===" % (name, len(fr)))
        print("%-16s %5s %9s %8s" % ("モデル", "N", tkey, "精度"))
        for p in fr:
            print("%-16s %5d %9.2f %8.4f" % (LABEL[p["model"]], p["res"], p[tkey], p["acc"]))
        cols = ["model", "res", "input_res", "acc", "gpu_ms", "e2e_ms"]
        if has_dep:
            cols.append("e2e_deploy_ms")
        out["pareto_%s" % tkey] = [{k: p[k] for k in cols} for p in fr]

    # ---- (3) 時間制約つきの最良精度 ----
    print("\n=== 表 C: 時間制約ごとの最良構成 ===")
    print("%9s | %s" % ("制約", " | ".join("%-28s" % tname[k] for k in tkeys)))
    budgets = {}
    for T in TIME_BUDGETS:
        row = {}
        cells = []
        for tkey in tkeys:
            ok = [p for p in pts if p[tkey] <= T]
            if ok:
                # ⛔⛔ 査読指摘 A-1a (2026-09-08): **ここは選択なので acc_sel で選ぶ**。
                #   以前は `max(ok, key=...p["acc"])` としており、**報告用の test 精度を
                #   最大化していた**。本文は「構成選択はすべて validation で行う」と書いて
                #   いるので実装と食い違っていた (表 D の精度目標側は元から acc_sel で正しい)。
                #   実測の影響 = 8 セル中 4 セルが変わる (33.3 ms の DINOv2-L N=208 -> 224、
                #   50 ms 配備と 100 ms の DINOv3-L N=224 -> 208)。10 ms の 2 例は不変。
                # ⭐ 同値時の優先規則も定める: 選択精度が同じなら**短い方**、それも同じなら
                #   **低い解像度**を採る (運用上ありがたい側へ倒す。丸め前の値で比較する)。
                b = max(ok, key=lambda p: (p["acc_sel"], -p[tkey], -p["res"]))
                row[tkey] = {"model": b["model"], "res": b["res"],
                             "acc": round(b["acc"], 4),
                             "acc_selected_on": round(b["acc_sel"], 4),
                             "gpu_ms": b["gpu_ms"], "e2e_ms": b["e2e_ms"],
                             **({"e2e_deploy_ms": b["e2e_deploy_ms"]} if has_dep else {})}
                cells.append("%s N=%d  %.4f (%.1f ms)" % (LABEL[b["model"]], b["res"], b["acc"], b[tkey]))
            else:
                row[tkey] = None
                cells.append("(該当なし)")
        budgets["%.1f" % T] = row
        print("%7.1fms | %s" % (T, " | ".join("%-28s" % c for c in cells)))
    out["best_under_time_budget"] = budgets

    # ---- (4) 精度目標つきの最短時間 ----
    print("\n=== 表 D: 精度目標ごとの最短時間構成 ===")
    print("%8s | %s" % ("目標精度", " | ".join("%-30s" % (tname[k] + " 最小") for k in tkeys)))
    targets = {}
    for A in ACC_TARGETS:
        row = {}
        cells = []
        for tkey in tkeys:
            ok = [p for p in pts if p["acc_sel"] >= A]   # 選択は acc_sel (A5)
            if ok:
                b = min(ok, key=lambda p: p[tkey])
                row[tkey] = {"model": b["model"], "res": b["res"],
                             "acc": round(b["acc"], 4),
                             "acc_selected_on": round(b["acc_sel"], 4),
                             "gpu_ms": b["gpu_ms"], "e2e_ms": b["e2e_ms"],
                             **({"e2e_deploy_ms": b["e2e_deploy_ms"]} if has_dep else {})}
                cells.append("%s N=%d  %.1f ms (%.4f)" % (LABEL[b["model"]], b["res"], b[tkey], b["acc"]))
            else:
                row[tkey] = None
                cells.append("(達成不可)")
        targets["%.2f" % A] = row
        print("%8.2f | %s" % (A, " | ".join("%-30s" % c for c in cells)))
    out["fastest_meeting_accuracy"] = targets

    # ---- (5) 判別スループット (実システムの構成に基づく) ----
    # 「1 カメラ視野内で最大何個体の飛翔体を扱えるか」に答えるための指標。
    # 判別は抽出された候補に対してのみ走るのでフレーム周期には縛られないが、
    # **7 カメラ分の判別が 1 台 (または 2 台) の計算機を共有する**ため、
    # 1 秒あたりに処理できる個体数には上限がある。
    ekey = "e2e_deploy_ms" if has_dep else "e2e_ms"
    print("\n=== 表 E: 判別スループット (1 ユニット = Orin Nano 1 台が 4 カメラを時分割) ===")
    print("     時間軸 = %s" % tname[ekey])
    print("%-16s %8s | %12s %12s" %
          ("モデル", "e2e[ms]", "c=1 個体/s/cam", "c=4 個体/s/cam"))
    thr = {}
    for m in MODELS:
        p = next((q for q in pts if q["model"] == m and q["res"] == 112), None)
        if not p:
            continue
        e = p[ekey]
        rec = {}
        for c in CLASSIFY_PER_BIRD:
            # Orin 1 台が毎秒 1000/e 回の判別をこなし、それを 4 カメラで分け合う。
            # 1 個体につき c 回の判別を要するので個体数は c で割る。
            rec["per%d" % c] = round(1000.0 / e / NCAM / c, 2)
        thr[m] = {"res": 112, "e2e_ms": p["e2e_ms"], "time_basis": ekey, ekey: e, **rec}
        print("%-16s %8.2f | %12.1f %12.1f" % (LABEL[m], e, rec["per1"], rec["per4"]))
    out["classification_throughput"] = {
        "note": ("1 カメラ視野内で扱える飛翔体数の上限。1 ユニット = 周辺 3 台 + 追跡 PTZ 1 台 = "
                 "4 カメラを Jetson Orin Nano 8GB 1 台が時分割する。360 度監視は 2 ユニット構成だが、"
                 "各ユニットが独立に 4 カメラを担当するので 1 カメラあたりの能力は変わらない。"
                 "1 個体あたりの判別回数は原則 1 回、危険度上昇時 (3/2/1 分後に回転半径の 2-3 倍圏内へ"
                 "到達予測) の追加を含めて最大 4 回。視野内の滞在時間を掛ければ同時個体数が得られる"),
        "n_cameras_per_unit": NCAM, "n_units_for_360deg": NUNIT,
        "classifications_per_bird": list(CLASSIFY_PER_BIRD),
        "unit": "birds/s/camera", "by_model_at_N112": thr,
    }

    # ---- (6) 2 経路の突き合わせ ----
    # 「N を下げても速くならない」という結論が、どちらの前処理経路を前提にするかで
    # 逆転することを、同一の GPU 実測値のうえで示す。
    if has_dep:
        print("\n=== 表 F: オフライン評価経路 対 配備経路 ===")
        print("%-16s %5s | %9s %9s %9s | %9s %9s" %
              ("モデル", "N", "GPU", "オフ前処理", "配備前処理", "オフ e2e", "配備 e2e"))
        cmp_rows = []
        for m in MODELS:
            for n in (224, 112, 80, 16):
                p = next((q for q in pts if q["model"] == m and q["res"] == n), None)
                if not p:
                    continue
                cmp_rows.append({k: p[k] for k in
                                 ("model", "res", "gpu_ms", "prep_ms", "prep_deploy_ms",
                                  "e2e_ms", "e2e_deploy_ms")})
                print("%-16s %5d | %9.3f %9.3f %9.3f | %9.3f %9.3f" %
                      (LABEL[m], n, p["gpu_ms"], p["prep_ms"], p["prep_deploy_ms"],
                       p["e2e_ms"], p["e2e_deploy_ms"]))

        # 代表モデルで N=224 -> 80 の短縮率を両経路で比較する
        red = {}
        for m in MODELS:
            a = next((q for q in pts if q["model"] == m and q["res"] == 224), None)
            b = next((q for q in pts if q["model"] == m and q["res"] == 80), None)
            if not (a and b):
                continue
            red[m] = {
                "offline_pct": round(100 * (1 - b["e2e_ms"] / a["e2e_ms"]), 1),
                "deploy_pct": round(100 * (1 - b["e2e_deploy_ms"] / a["e2e_deploy_ms"]), 1),
                "offline_224_ms": a["e2e_ms"], "offline_80_ms": b["e2e_ms"],
                "deploy_224_ms": a["e2e_deploy_ms"], "deploy_80_ms": b["e2e_deploy_ms"],
            }
        print("\n  N=224 -> 80 の e2e 短縮率")
        for m, v in red.items():
            print("    %-16s オフライン %5.1f%% (%6.2f -> %6.2f) / 配備 %5.1f%% (%6.2f -> %6.2f)" %
                  (LABEL[m], v["offline_pct"], v["offline_224_ms"], v["offline_80_ms"],
                   v["deploy_pct"], v["deploy_224_ms"], v["deploy_80_ms"]))

        fastest_off = min(pts, key=lambda p: p["e2e_ms"])
        fastest_dep = min(pts, key=lambda p: p["e2e_deploy_ms"])
        print("\n  最速構成: オフライン %s N=%d %.2f ms / 配備 %s N=%d %.2f ms" %
              (LABEL[fastest_off["model"]], fastest_off["res"], fastest_off["e2e_ms"],
               LABEL[fastest_dep["model"]], fastest_dep["res"], fastest_dep["e2e_deploy_ms"]))
        out["path_comparison"] = {
            "note": ("同一の GPU 実測値のうえで前処理経路だけを替えた比較。"
                     "⚠️bbox と N が一致する水準ではリサイズが不要になり配備側が下振れするが、"
                     "補間はしていない (短縮率を小さく見せる保守側の扱い)"),
            "deploy_bbox_px": DEPLOY_BBOX,
            "rows": cmp_rows,
            "reduction_224_to_80": red,
            "fastest_offline": {k: fastest_off[k] for k in ("model", "res", "e2e_ms")},
            "fastest_deploy": {k: fastest_dep[k] for k in ("model", "res", "e2e_deploy_ms")},
        }

    if args.out:
        dst = args.out
    elif is_trt10():
        dst = os.path.join(R, "optimal_n_%s.json" % SRC.replace("-", "_"))
    else:
        dst = os.path.join(R, "optimal_n.json")
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print("\n[saved] %s" % dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
