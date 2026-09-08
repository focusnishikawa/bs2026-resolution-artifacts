#!/usr/bin/env python3
"""Phase 5: 全フェーズの結果を統合し、論文の主張に対応する表を生成する.

入力 (すべて Mac 側 results/ に集約済み):
  results/summary_v2.json          T1 精度 (条件 A/B x 6 モデル x 14 解像度 x 3 seed)
  results/orin/results/*.json      Orin latency (4 モデル x 14 解像度)
  results/orin/results_h200/*.json H200 latency (6 モデル x 14 解像度)
  results/orin/results/fp16_*.json FP16 精度検証
  results/v6/v6_res14_summary.json v6 カスケード (3 系統 x 14 解像度)

出力: results/final_tables.json と標準出力の表

⭐ データ源 (--src) — 2026-08-31 に追加
  JetPack 6.2 化で TensorRT が 8.5.2 -> 10.3.0 になり既存エンジンが全滅したため、
  Orin の測定を全部やり直した。旧データの再現性を壊さずに新データも読めるよう、
  collect_vitl_edge.py と同じ流儀でデータ源を選べるようにしてある。

  legacy      JP5.1.2 / TRT 8.5.2  results{,_v2,_v30}/ + orin_vitl/vitl_edge_summary.json
  trt10-15w   JP6.2   / TRT 10.3   results_trt10_15w/  + ..._summary_trt10_15w.json
  trt10-maxn  JP6.2   / TRT 10.3   results_trt10_maxn/ + ..._summary_trt10_maxn.json

  ⚠️ **trt10 では旧測定を混ぜない。** legacy は results -> results_v2 -> results_v30 と
     新しい測定で順に上書きする積み上げだが、TensorRT 版が違う値を積むと
     「新環境 GPU + 旧環境 GPU」の混合になる。trt10 では積まず、**欠損は欠損として報告する**。
     ⭐ FP16 飽和で壊れたエンジンは計算が消えるぶん速く見えるので、latency だけでは気づけない。

usage:
    python3 train/analyze_all.py                    # 旧環境 (既定・出力名も従来どおり)
    python3 train/analyze_all.py --src trt10-maxn   # JP6.2 / TRT 10.3 / MAXN_SUPER
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]
# ViT-L は分割チェーンなので通常のスイープに乗らない (collect_vitl_edge.py の担当)
CNN_MODELS = ["mnv4", "effb0", "resnet50", "vit_small"]

# ---- データ源 (main() で --src により設定する。既定は旧環境で振る舞い不変) ----
SRC = "legacy"
# TRT 10.3 の CNN 測定 JSON の置き場 (measure_trt10_mode.sh の OUT を回収したもの)
TRT10_DIRS = {"trt10-15w": "results_trt10_15w", "trt10-maxn": "results_trt10_maxn"}
SRC_ENV = {"legacy": {"jetpack": "5.1.2", "l4t": "R35.4.1", "trt": "8.5.2", "power_mode": "15W"},
           "trt10-15w": {"jetpack": "6.2", "l4t": "R36.4.3", "trt": "10.3.0", "power_mode": "15W"},
           "trt10-maxn": {"jetpack": "6.2", "l4t": "R36.4.3", "trt": "10.3.0",
                          "power_mode": "MAXN_SUPER"}}
N_EXPECTED = len(CNN_MODELS) * len(RES)  # = 56 構成


def is_trt10():
    return SRC in TRT10_DIRS


def vitl_summary_path():
    """ViT-L 集計の置き場。collect_vitl_edge.py:545-549 の出力先と一致させる"""
    if is_trt10():
        return os.path.join(R, "orin_vitl", "vitl_edge_summary_%s.json" % SRC.replace("-", "_"))
    return os.path.join(R, "orin_vitl", "vitl_edge_summary.json")


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load_latency(dirname, key):
    """Orin / H200 の latency を {model: {res: ms}} で返す。"""
    out = {}
    for f in glob.glob(os.path.join(R, "orin", dirname, "*.json")):
        b = os.path.basename(f)
        if not re.match(r"^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r\d+\.json$", b):
            continue
        j = json.load(open(f))
        v = j.get(key)
        if isinstance(v, dict):
            v = v.get("mean")
        if v is not None:
            out.setdefault(j["model"], {})[j["res"]] = v
    return out


def load_latency_stats(dirname):
    """30 回測定の統計量をまるごと返す {model: {res: {mean_ms, sd_ms, ...}}}.

    3 回測定では中央値しか報告できなかったが、30 回あれば標準偏差と信頼区間を出せる。
    論文の表はここから「平均 ± 標準偏差」で書く。
    """
    out = {}
    for f in glob.glob(os.path.join(R, "orin", dirname, "*.json")):
        b = os.path.basename(f)
        if not re.match(r"^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r\d+\.json$", b):
            continue
        j = json.load(open(f))
        if "median_ms" not in j:
            continue
        out.setdefault(j["model"], {})[j["res"]] = {
            k: j[k] for k in ("mean_ms", "sd_ms", "median_ms", "min_ms", "max_ms",
                              "ci95_halfwidth_ms", "cv_pct", "reps", "engine_MB") if k in j}
    return out


def main():
    global SRC
    ap = argparse.ArgumentParser(description="全フェーズの結果を統合し論文の表を生成する")
    ap.add_argument("--src", default="legacy", choices=["legacy"] + sorted(TRT10_DIRS),
                    help="データ源 (既定 legacy = JetPack 5.1.2 / TRT 8.5.2)")
    ap.add_argument("--out", default=None,
                    help="出力先 JSON (既定はデータ源ごとの名前。legacy は従来どおり)")
    args = ap.parse_args()
    SRC = args.src

    acc = load_json(os.path.join(R, "summary_v2.json"))
    missing = []
    if is_trt10():
        # ⚠️ 旧測定を積まない (TensorRT 版が違う値は混ぜられない)。
        #    欠損があっても旧値で埋めず、欠損として報告する。
        d = TRT10_DIRS[SRC]
        orin = load_latency(d, "median_ms")
        orin_stats = load_latency_stats(d)
        n = sum(len(v) for v in orin.values())
        e = SRC_ENV[SRC]
        print("[orin] **TRT 10.3 の測定 (%s) を採用**: %d/%d 構成 "
              "(JetPack %s / L4T %s / %s)"
              % (SRC, n, N_EXPECTED, e["jetpack"], e["l4t"], e["power_mode"]))
        missing = ["%s_r%d" % (m, r) for m in CNN_MODELS for r in RES
                   if r not in orin.get(m, {})]
        if missing:
            print("  ⚠️ **欠損 %d 件** (旧値で埋めていない): %s"
                  % (len(missing), ", ".join(missing)))
    else:
        # Orin は当初 1 構成 1 回計測だったが、ViT-S/16 の N=64 だけ FP16 経路が選ばれず
        # FP32 のままビルドされていた (engine 86.3 MB / 他水準 44 MB、latency +39%)。
        # 再ビルドのうえ全 56 構成を H200 と同じ 3 回測定へ揃え直したものが results_v2。
        # さらに 8/20 に **30 回測定** (results_v30) へ拡張した。ばらつきを報告できるのはこれだけ。
        # 原本 (results/) は残し、新しい測定があるものから順に上書きする。
        orin = load_latency("results", "gpu_compute_ms")
        orin_v2 = load_latency("results_v2", "gpu_compute_mean_ms_median_of_reps")
        if orin_v2:
            n = sum(len(v) for v in orin_v2.values())
            print("[orin] 3 回測定版 (results_v2) を採用: %d 構成" % n)
            for m, d in orin_v2.items():
                orin.setdefault(m, {}).update(d)
        orin_v30 = load_latency("results_v30", "median_ms")
        orin_stats = load_latency_stats("results_v30")
        if orin_v30:
            n = sum(len(v) for v in orin_v30.values())
            print("[orin] **30 回測定版 (results_v30) を採用**: %d 構成" % n)
            for m, d in orin_v30.items():
                orin.setdefault(m, {}).update(d)
    h200 = load_latency("results_h200", "gpu_compute_mean_ms_median_of_reps")
    v6 = load_json(os.path.join(R, "v6", "v6_res14_summary.json"))
    # 表 4 の精度もデータ源に従う。trt10 では preds_trt10/ から再集計したものを読む
    # (作り方は train/fp16_accuracy_trt10.py。旧環境の値と混ぜないため名前を分けてある)
    fp16_name = "fp16_accuracy_trt10.json" if is_trt10() else "fp16_accuracy.json"
    fp16_path = os.path.join(R, "orin", "results", fp16_name)
    fp16 = load_json(fp16_path)
    tables = {}
    if is_trt10():
        # 旧環境の値と取り違えないよう、出典をファイルの先頭に残す
        tables["source"] = SRC
        tables["measurement_env"] = SRC_ENV[SRC]
        tables["orin_coverage"] = {"n": sum(len(v) for v in orin.values()),
                                   "expected": N_EXPECTED, "missing": missing,
                                   "complete": not missing}

    # ---- 表 1: 所要画素数 (224 比 -5 pt 以内を保てる最小 N) ----
    print("=== 表 1: 所要画素数 (224 比 -5 pt 以内, T1 猛禽 6 種, 3 seed 平均) ===")
    print("%-12s %10s %10s %14s" % ("model", "条件A", "条件B", "適応の効果"))
    t1 = {}
    for m in MODELS:
        row = {}
        for cond in ("A", "B"):
            base = acc["models"][m]["224"][cond]["mean"]
            ok = [r for r in RES if (acc["models"][m][str(r)][cond]["mean"] - base) * 100 >= -5.0]
            row[cond] = min(ok) if ok else None
        eff = "改善" if row["B"] < row["A"] else ("悪化" if row["B"] > row["A"] else "同等")
        print("%-12s %10s %10s %14s" % (m, row["A"], row["B"], eff))
        t1[m] = row
    tables["pixel_requirement"] = t1

    # ---- 表 2: Orin 実測 latency と解像度スケーリング ----
    print("\n=== 表 2: Orin Nano FP16 latency (ms) と 224/16 比 ===")
    print("%-12s" % "model" + "".join("%8d" % r for r in [16, 64, 112, 160, 224]) + "%10s" % "r224/r16")
    t2 = {}
    for m in MODELS:
        if m not in orin:
            print("%-12s%s" % (m, "   (Orin では実行不可)"))
            continue
        vals = [orin[m].get(r) for r in [16, 64, 112, 160, 224]]
        # ⚠️ 測定が途中だと N=224 / N=16 が欠けうる (trt10 の部分データで実際に KeyError になった)。
        #    legacy は 56 構成そろっているので値は変わらない。
        v224, v16 = orin[m].get(224), orin[m].get(16)
        ratio = (v224 / v16) if (v224 and v16) else None
        print("%-12s" % m + "".join("%8.3f" % v if v else "       -" for v in vals)
              + ("%10.2f" % ratio if ratio is not None else "%10s" % "-"))
        t2[m] = {"latency": orin[m],
                 "ratio_224_16": round(ratio, 3) if ratio is not None else None}
        # 30 回測定があれば統計量も持たせる (論文の表は平均 +- 標準偏差で書く)
        if m in orin_stats:
            t2[m]["stats"] = orin_stats[m]
    tables["orin_latency"] = t2
    if orin_stats:
        cvs = [s["cv_pct"] for d in orin_stats.values() for s in d.values() if "cv_pct" in s]
        n30 = sum(len(d) for d in orin_stats.values())
        if cvs:
            print("  [30 回測定] %d 構成、変動係数 %.2f-%.2f%% (中央値 %.2f%%)"
                  % (n30, min(cvs), max(cvs), sorted(cvs)[len(cvs) // 2]))

    # ---- 表 3: 精度とレイテンシのパレート (Orin) ----
    print("\n=== 表 3: パレート候補 (Orin 実測。精度は条件 B の 3 seed 平均) ===")
    print("%-12s %5s %10s %10s %12s" % ("model", "N", "acc", "latency", "acc/ms"))
    pareto = []
    for m in MODELS:
        if m not in orin:
            continue
        for r in RES:
            a = acc["models"][m][str(r)]["B"]["mean"]
            l = orin[m].get(r)
            if l:
                pareto.append((m, r, a, l))
    # パレート最適 (これより速くて高精度な点が無いもの)
    front = [p for p in pareto if not any(q[3] <= p[3] and q[2] >= p[2] and q != p for q in pareto)]
    for m, r, a, l in sorted(front, key=lambda x: x[3]):
        print("%-12s %5d %10.4f %10.3f %12.1f" % (m, r, a, l, a / l * 100))
    tables["pareto_front"] = [{"model": m, "res": r, "acc": a, "latency_ms": l} for m, r, a, l in front]

    # ---- 表 4: FP16 の安全性 ----
    if fp16:
        print("\n=== 表 4: Orin FP16 の精度劣化 (全数 n=1882) ===")
        # ⚠️ latency だけ差し替えて精度を旧環境のまま載せると出典が混ざるので、
        #    どちらを読んだのかを必ず示す。
        #    ⚠️ legacy の JSON にはキーを足さない (バイト一致の回帰を壊さないため)
        print("  出典: %s / 参照: サーバ FP32 seed 42 (results/T1_condB_s42/preds)"
              % os.path.basename(fp16_path))
        if is_trt10():
            tables["fp16_source"] = "trt10 (JetPack 6.2 / TRT 10.3, preds_trt10/ から再集計)"
            tables["fp16_reference"] = "server PyTorch FP32 (results/T1_condB_s42/preds)"
        print("%-18s %10s %10s %12s" % ("config", "FP32", "FP16", "argmax一致"))
        for r in sorted(fp16, key=lambda x: (x["model"], x["res"])):
            print("%-18s %10.4f %10.4f %11.2f%%" % (r["tag"], r["fp32_acc"], r["fp16_acc"],
                                                    r["argmax_agreement"] * 100))
        tables["fp16"] = fp16

    # ---- 表 5: v6 カスケード ----
    if v6:
        print("\n=== 表 5: v6 IUCN カスケード (アラート precision) ===")
        print("%-12s" % "system" + "".join("%8d" % r for r in [32, 48, 64, 96, 128, 192]))
        for s in ("condA_flat", "condA_s34", "condB_fix"):
            if s not in v6:
                continue
            row = ""
            for r in [32, 48, 64, 96, 128, 192]:
                v = v6[s].get(str(r), {}).get("alert_precision")
                row += ("%8.4f" % v) if v is not None else "       -"
            print("%-12s" % s + row)
        tables["v6"] = {s: v6[s] for s in v6}

    # ---- 表 6: エッジ vs サーバ ----
    print("\n=== 表 6: Orin vs H200 (N=224, ms) ===")
    print("%-12s %10s %10s %10s" % ("model", "Orin", "H200", "比"))
    t6 = {}
    # ViT-L は分割チェーンなので通常のスイープに乗らない。8/20 に全 14 水準を測ったので
    # N=224 の値をここへ差し込む (両モデルとも配備構成の 3 回測定中央値)
    vitl_json = vitl_summary_path()
    vitl_sweep = {}
    if os.path.exists(vitl_json):
        _v = json.load(open(vitl_json))
        for m, key in (("dinov2_l", "resolution_sweep_dinov2_l"), ("dinov3_l", "resolution_sweep_dinov3_l")):
            if key in _v:
                vitl_sweep[m] = _v[key]["by_res"]
    for m in MODELS:
        o = orin.get(m, {}).get(224)
        h = h200.get(m, {}).get(224)
        if o is None and m in vitl_sweep:
            o = vitl_sweep[m]["224"]["latency_ms"]
        if o and h:
            print("%-12s %10.3f %10.3f %9.1fx" % (m, o, h, o / h))
            t6[m] = {"orin": o, "h200": h, "ratio": round(o / h, 2)}
        elif h:
            print("%-12s %10s %10.3f %10s" % (m, "N=224 未測定", h, "-"))
            t6[m] = {"orin": None, "h200": h,
                     "note": "N=224 は未測定。r32 の実測は tables['orin_vitl'] を見る"}
    tables["orin_vs_h200"] = t6

    # ---- 表 7: ViT-L のエッジ実測 (r32 のみ。8/18-19 の追加実験) ----
    vitl = vitl_summary_path()
    if os.path.exists(vitl):
        v = json.load(open(vitl))
        tables["orin_vitl"] = v
        print("\n=== 表 7: ViT-L の Orin 実測 (N=32、4 分割チェーン) ===")
        print("%-12s %10s %10s %8s %10s %10s" % ("model", "Orin", "H200", "比", "acc", "一致"))
        for m, r in v["models"].items():
            print("%-12s %9.3f %9.3f %7.1fx %10.4f %9.2f%%"
                  % (m, r["latency_ms"], r["h200_r32_ms"], r["edge_server_ratio"],
                     r.get("orin_fp16_acc", float("nan")),
                     r.get("argmax_agreement", float("nan")) * 100))

    if args.out:
        out = args.out
    elif is_trt10():
        out = os.path.join(R, "final_tables_%s.json" % SRC.replace("-", "_"))
    else:
        out = os.path.join(R, "final_tables.json")
    json.dump(tables, open(out, "w"), indent=2, ensure_ascii=False)
    print("\n[saved] %s" % out)
    if is_trt10():
        e = SRC_ENV[SRC]
        print("  データ源 %s (JetPack %s / L4T %s / TRT %s / %s)"
              % (SRC, e["jetpack"], e["l4t"], e["trt"], e["power_mode"]))
        if missing:
            print("  ⚠️ **欠損 %d 件のまま集計した**。表 2・表 3・パレートは不完全である"
                  % len(missing))


if __name__ == "__main__":
    main()
