#!/usr/bin/env python3
"""ViT-S/16 を **FP32 のまま配備する構成**を候補へ加えると推奨がどう変わるかを出す.

目的:
  査読指摘 A1 で報告値を配備エンジン (Orin FP16) の実測へ差し替えた結果、ViT-S/16 は
  FP16 化の劣化で配備候補からほぼ外れた。一方 FP32 でビルドすれば精度はサーバとほぼ同じに
  戻る代わりに GPU 時間が約 2.5 倍になる (N=224: 2.359 -> 5.920 ms)。
  そこで **ViT-S/16 FP32 の 14 構成を候補集合に足したら、目標精度ごとの最短構成・
  時間予算ごとの最良構成・パレートフロントが動くのか**を、既存の選択規則のまま計算する。

入出力:
  in   results/summary_deploy_fp32.json     train/aggregate_fp32.py の出力 (FP32 の val/test/latency)
       results/summary_deploy_val.json      配備 FP16 の val   (選択に使う。A5)
       results/summary_deploy.json          配備 FP16 の test  (報告に使う。A1)
       results/summary_val.json             サーバ FP32 の val (選択側の落とし先・(a) の比較対象)
       results/summary_v2.json              サーバ FP32 の test((a) の比較対象。optimal_n.load の既定)
       results/final_tables_trt10_maxn.json Orin latency (FP16)          … optimal_n.tables_path()
       results/orin/results_prep_trt10/prep_timing_{B,deploy}_maxn.json  … optimal_n.prep_path()
       results/target_sweep.json            回帰ガードの基準 (deployed 側 winner 176 点)
       results/optimal_n_a1.json            回帰ガードの基準 (時間予算 4 点・パレート 3 集合)
  out  results/vits_fp32_candidacy.json

規則の出典 (すべて optimal_n.py / target_sweep.py の実装に従う。ここで新しい規則は作らない):
  latency の点値      analyze_all.py:122  load_latency(d, "median_ms")  -> median を使う
                      optimal_n.py:196-200 load() が tables["orin_latency"][m]["latency"] を引く
  e2e の組み立て      optimal_n.py:224-234 load()
                        prep_ms        = prep_timing_B_maxn.json      by_res[N].total.mean
                        prep_deploy_ms = prep_timing_deploy_maxn.json per_bird[224][N].total_cv
                        e2e_ms         = round(prep_ms + gpu_ms, 3)
                        e2e_deploy_ms  = round(prep_deploy_ms + gpu_ms, 3)
                      ⭐ 前処理は**演算精度に依存しない** (bbox 切り出し・リサイズ・正規化の CPU 側)
                         ので、FP32 構成にも vit_small と同じ前処理値を使う。
  時間予算の選択      optimal_n.py:403-413  max(ok, key=(acc_sel, -time, -res))  ok = time <= T
                      ⚠️ **選択は acc_sel (validation)**、報告は acc (test)。
  目標精度の選択      optimal_n.py:435-438  min(ok, key=time)          ok = acc_sel >= A
                      target_sweep.py:45-61 pick() は同値時 (time, res) の辞書順で最小を採る
                      ⭐ 本スクリプトの目標側は **target_sweep.pick() をそのまま呼んで**基準を作る。
  パレート            optimal_n.py:239-249  pareto() … **acc_sel** で支配関係を判定する

シード集合:
  vit_small_fp32 は 30 本 (42-71)。既存の「ViT-L 級 29 本・他 30 本」の扱いは変えない。

usage:
  python3 train/vits_fp32_candidacy.py [--out results/vits_fp32_candidacy.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import optimal_n as O                                              # noqa: E402
import target_sweep as TS                                          # noqa: E402

R = os.path.join(ROOT, "results")
SRC = "trt10-maxn"
FP32_MODEL = "vit_small_fp32"
FP32_BASE = "vit_small"          # 前処理・ラベル・比較の基になる FP16 側モデル
LAT_MODE = "maxn"                # optimal_n の --src trt10-maxn と同じ電力モード
TKEY = "e2e_deploy_ms"           # target_sweep.py の既定 (配備経路)
LABEL = dict(O.LABEL, **{FP32_MODEL: "ViT-S/16 (FP32)"})


def fmt(p):
    return "%s N=%d" % (LABEL[p["model"]], p["res"])


# ---------------------------------------------------------------- 候補点の構築
def build_fp32_points(fp32, prep_b, prep_dep):
    """FP32 の 14 構成を optimal_n.load() と同じ形のレコードにする"""
    pts = []
    for n in O.RES:
        k = str(n)
        g = fp32["latency_fp32"][LAT_MODE][k]["median_ms"]
        pre = prep_b[k]["total"]["mean"]
        pd = prep_dep[str(O.DEPLOY_BBOX)][k]["total_cv"]
        pd_alt = prep_dep[str(O.DEPLOY_BBOX_ALT)][k]["total_cv"]
        a = fp32["splits"]["test"][FP32_MODEL][k]["B"]
        v = fp32["splits"]["val"][FP32_MODEL][k]["B"]
        pts.append({
            "model": FP32_MODEL, "res": n, "input_res": O.input_res(FP32_BASE, n),
            "acc": a["mean"], "acc_sd": a["std"], "acc_source": "deployed_fp32",
            "acc_sel": v["mean"], "acc_sel_sd": v["std"], "acc_sel_source": "deployed_fp32",
            "n_seeds": a["n_seed"],
            "gpu_ms": round(g, 3), "prep_ms": round(pre, 3),
            "e2e_ms": round(pre + g, 3),
            "prep_deploy_ms": round(pd, 3), "e2e_deploy_ms": round(pd + g, 3),
            "e2e_deploy_bbox%d_ms" % O.DEPLOY_BBOX_ALT: round(pd_alt + g, 3),
        })
    return pts


# ---------------------------------------------------------------- 選択規則
def pick_target(pts, A, tkey):
    """目標 A を満たす最短構成 (optimal_n.py:435-438 / target_sweep.pick と同一規則)"""
    ok = [p for p in pts if p["acc_sel"] >= A]
    if not ok:
        return None
    return min(ok, key=lambda p: (p[tkey], p["res"]))


def pick_budget(pts, T, tkey):
    """予算 T 以内の最良構成 (optimal_n.py:403-413 と同一規則)"""
    ok = [p for p in pts if p[tkey] <= T]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["acc_sel"], -p[tkey], -p["res"]))


def budget_row(b):
    """optimal_n.py:414-418 が書く形"""
    return {"model": b["model"], "res": b["res"], "acc": round(b["acc"], 4),
            "acc_selected_on": round(b["acc_sel"], 4),
            "gpu_ms": b["gpu_ms"], "e2e_ms": b["e2e_ms"],
            "e2e_deploy_ms": b["e2e_deploy_ms"]}


def sweep_row(p, tkey):
    """target_sweep.pick() が書く形 (fallback は acc_sel_source から復元する)"""
    return {"model": p["model"], "res": p["res"], "val": p["acc_sel"],
            "fallback": p["acc_sel_source"] == "server_fp32",
            "time_ms": p[tkey], "test": p["acc"]}


def targets(lo, hi, step):
    """target_sweep.py:98-107 と同じ刻み方 (浮動小数の積み方も合わせる)"""
    out, t = [], lo
    while t <= hi + 1e-9:
        out.append(t)
        t += step
    return out


def dominates(q, p, tkey):
    """optimal_n.pareto() の支配判定 (acc_sel で見る)"""
    return ((q[tkey] <= p[tkey] and q["acc_sel"] >= p["acc_sel"]) and
            (q[tkey] < p[tkey] or q["acc_sel"] > p["acc_sel"]))


# ---------------------------------------------------------------- 回帰ガード
def guard_sweep(pts, dv, sv, ts_json, tkey, step, lo, hi):
    """FP32 を加える前に、deployed 側 winner 176 点が target_sweep.json と完全一致するか"""
    rows = ts_json["rows"]
    ts = targets(lo, hi, step)
    assert len(rows) == len(ts) == 176, "目標点数が %d / %d (期待 176)" % (len(rows), len(ts))
    for t, ref in zip(ts, rows):
        assert round(t, 4) == ref["target"], "目標がずれた %.4f / %.4f" % (t, ref["target"])
        got = TS.pick(pts, dv, sv, t, tkey)             # 実装そのものを呼ぶ
        assert got == ref["deploy"], \
            "目標 %.3f の deployed winner が不一致\n  再計算 %s\n  基準   %s" % (t, got, ref["deploy"])
        # 本スクリプトの一般化版 (acc_sel を直接見る) が同じ結果を返すことも確かめる。
        # ⚠️ これが崩れると FP32 を足したあとの計算が基準と別規則になってしまう。
        mine = pick_target(pts, t, tkey)
        assert sweep_row(mine, tkey) == ref["deploy"], \
            "目標 %.3f で一般化版がずれた: %s" % (t, sweep_row(mine, tkey))
    return len(ts)


def guard_budget(pts, on_json):
    """時間予算 4 点 x 時間軸 3 種が optimal_n_a1.json と一致するか"""
    ref = on_json["best_under_time_budget"]
    n = 0
    for T in O.TIME_BUDGETS:
        for tkey in ("gpu_ms", "e2e_ms", "e2e_deploy_ms"):
            b = pick_budget(pts, T, tkey)
            got = budget_row(b) if b else None
            want = ref["%.1f" % T][tkey]
            assert got == want, \
                "予算 %.1f ms / %s が不一致\n  再計算 %s\n  基準   %s" % (T, tkey, got, want)
            n += 1
    return n


def guard_pareto(pts, on_json):
    """パレート 3 集合が optimal_n_a1.json と一致するか"""
    n = 0
    for tkey in ("gpu_ms", "e2e_ms", "e2e_deploy_ms"):
        fr = O.pareto(pts, tkey)
        got = [(p["model"], p["res"]) for p in fr]
        want = [(p["model"], p["res"]) for p in on_json["pareto_%s" % tkey]]
        assert got == want, "パレート (%s) が不一致\n  再計算 %s\n  基準   %s" % (tkey, got, want)
        n += len(got)
    return n


# ---------------------------------------------------------------- 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.001)
    ap.add_argument("--lo", type=float, default=0.80)
    ap.add_argument("--hi", type=float, default=0.975)
    ap.add_argument("--out", default=os.path.join(R, "vits_fp32_candidacy.json"))
    args = ap.parse_args()

    O.SRC = SRC
    fp32 = json.load(open(os.path.join(R, "summary_deploy_fp32.json")))
    sv = json.load(open(os.path.join(R, "summary_val.json")))        # サーバ FP32 の val
    s_test = json.load(open(os.path.join(R, "summary_v2.json")))     # サーバ FP32 の test
    dv = json.load(open(os.path.join(R, "summary_deploy_val.json")))
    ts_json = json.load(open(os.path.join(R, "target_sweep.json")))
    on_json = json.load(open(os.path.join(R, "optimal_n_a1.json")))

    # 既存の候補集合 (target_sweep.py:90-92 と同じ引数)
    base, _, has_dep = O.load(select_acc_path=os.path.join(R, "summary_deploy_val.json"),
                              report_acc_path=os.path.join(R, "summary_deploy.json"),
                              select_fallback_path=os.path.join(R, "summary_val.json"))
    assert has_dep, "配備経路の前処理が読めていない"
    print("\n[候補] 既存 %d 点 (%s)" % (len(base), SRC))

    # ---- 回帰ガード (FP32 を足す前) ----
    n_sw = guard_sweep(base, dv, sv, ts_json, TKEY, args.step, args.lo, args.hi)
    n_bg = guard_budget(base, on_json)
    n_pf = guard_pareto(base, on_json)
    print("[回帰ガード] target_sweep.json の deployed winner %d/%d 点 完全一致" % (n_sw, n_sw))
    print("[回帰ガード] optimal_n_a1.json の時間予算 %d セル (予算 4 x 時間軸 3) 完全一致" % n_bg)
    print("[回帰ガード] optimal_n_a1.json のパレート 3 集合 (計 %d 点) 完全一致" % n_pf)

    # ---- FP32 候補の構築 ----
    prep_b = json.load(open(O.prep_path("B")))["by_res"]
    prep_dep = json.load(open(O.prep_path("deploy")))["per_bird"]
    fps = build_fp32_points(fp32, prep_b, prep_dep)
    # 前処理は精度に依存しないので、FP16 の vit_small と同じ値でなければならない
    for p in fps:
        q = next(x for x in base if x["model"] == FP32_BASE and x["res"] == p["res"])
        assert (p["prep_ms"], p["prep_deploy_ms"]) == (q["prep_ms"], q["prep_deploy_ms"]), \
            "N=%d の前処理が vit_small と違う" % p["res"]
        assert p["n_seeds"] == 30, "N=%d のシード数が %s" % (p["res"], p["n_seeds"])
    both = base + fps
    print("[候補] FP32 を足して %d 点 (+%d)" % (len(both), len(fps)))

    out = {
        "note": ("ViT-S/16 を **FP32 のまま配備する** 14 構成を候補へ加えたときの推奨の変化。"
                 "選択規則・時間の組み立ては optimal_n.py / target_sweep.py の実装に従い、"
                 "FP32 を加えない状態で target_sweep.json / optimal_n_a1.json を"
                 "完全に再現できることを確かめてから足している"),
        "src": SRC, "tkey": TKEY, "latency_mode": LAT_MODE,
        "fp32_model_key": FP32_MODEL,
        "sources": {"fp32": "results/summary_deploy_fp32.json",
                    "deploy_val": "results/summary_deploy_val.json",
                    "deploy_test": "results/summary_deploy.json",
                    "server_val": "results/summary_val.json",
                    "server_test": "results/summary_v2.json",
                    "latency_fp16": os.path.basename(O.tables_path()),
                    "prep_offline": os.path.basename(O.prep_path("B")),
                    "prep_deploy": os.path.basename(O.prep_path("deploy")),
                    "sweep_baseline": "results/target_sweep.json",
                    "budget_baseline": "results/optimal_n_a1.json"},
        "n_points_base": len(base), "n_points_with_fp32": len(both),
        "regression_guard": {"target_sweep_points": n_sw,
                             "budget_cells": n_bg,
                             "pareto_points": n_pf,
                             "all_match": True},
        "fp32_points": fps,
    }

    # ---- (a) FP32 配備 - サーバ FP32 の差 ----
    print("\n=== (a) ViT-S/16 FP32: 配備 (Orin) - サーバ (PyTorch) [pt] ===")
    print("%5s | %8s %8s %8s | %8s %8s %8s" %
          ("N", "val配備", "valサーバ", "差", "test配備", "testサーバ", "差"))
    diffs = {}
    for n in O.RES:
        k = str(n)
        dvl = fp32["splits"]["val"][FP32_MODEL][k]["B"]
        dts = fp32["splits"]["test"][FP32_MODEL][k]["B"]
        svl = sv["models"][FP32_BASE][k]["B"]
        sts = s_test["models"][FP32_BASE][k]["B"]
        # val はどちらも seed 42-71 の 30 本 (同一シード集合)
        assert dvl["n_seed"] == svl["n_seed"] == 30, "N=%d の val シード数が違う" % n
        assert dts["n_seed"] == sts["n_seed"] == 30, "N=%d の test シード数が違う" % n
        d = {"val_deploy_fp32": round(dvl["mean"], 4), "val_server_fp32": round(svl["mean"], 4),
             "val_diff_pt": round(100 * (dvl["mean"] - svl["mean"]), 3),
             "test_deploy_fp32": round(dts["mean"], 4), "test_server_fp32": round(sts["mean"], 4),
             "test_diff_pt": round(100 * (dts["mean"] - sts["mean"]), 3),
             "n_seed": dvl["n_seed"]}
        # 参考: 同じ N の FP16 配備との差 (FP32 化で戻る幅)
        fp16v = dv["models"][FP32_BASE][k]["B"]["mean"]
        d["val_deploy_fp16"] = round(fp16v, 4)
        d["val_fp32_minus_fp16_pt"] = round(100 * (dvl["mean"] - fp16v), 3)
        diffs[k] = d
        print("%5d | %8.4f %8.4f %+8.2f | %8.4f %8.4f %+8.2f" %
              (n, d["val_deploy_fp32"], d["val_server_fp32"], d["val_diff_pt"],
               d["test_deploy_fp32"], d["test_server_fp32"], d["test_diff_pt"]))
    vr = [v["val_diff_pt"] for v in diffs.values()]
    tr = [v["test_diff_pt"] for v in diffs.values()]
    print("  val 差の範囲 %+.2f .. %+.2f pt / test 差の範囲 %+.2f .. %+.2f pt"
          % (min(vr), max(vr), min(tr), max(tr)))
    out["a_deploy_minus_server"] = {
        "note": ("配備 FP32 (Orin TRT10) - サーバ FP32 (PyTorch)。val はどちらも seed 42-71 の "
                 "30 本で同一シード集合。参考として同じ N の配備 FP16 (val) との差も入れた"),
        "unit": "pt", "by_res": diffs,
        "val_diff_pt_range": [min(vr), max(vr)], "test_diff_pt_range": [min(tr), max(tr)]}

    # ---- (b) 目標精度ごとの最短構成が変わる点 ----
    ts = targets(args.lo, args.hi, args.step)
    rows, changed = [], []
    for t in ts:
        a = pick_target(base, t, TKEY)
        b = pick_target(both, t, TKEY)
        same = bool(a and b and (a["model"], a["res"]) == (b["model"], b["res"]))
        rec = {"target": round(t, 4),
               "before": sweep_row(a, TKEY) if a else None,
               "after": sweep_row(b, TKEY) if b else None,
               "same": same}
        rows.append(rec)
        if not same:
            changed.append(rec)
    spans = []
    cur = None
    for rec in rows:
        if not rec["same"] and rec["before"] and rec["after"]:
            key = ((rec["before"]["model"], rec["before"]["res"]),
                   (rec["after"]["model"], rec["after"]["res"]))
            if cur and cur["_key"] == key:
                cur["to"] = rec["target"]
            else:
                if cur:
                    spans.append(cur)
                cur = {"_key": key, "from": rec["target"], "to": rec["target"],
                       "before": rec["before"], "after": rec["after"]}
        elif cur:
            spans.append(cur); cur = None
    if cur:
        spans.append(cur)
    for s in spans:
        s.pop("_key")
    print("\n=== (b) 目標精度ごとの最短構成: FP32 候補追加で変わる目標 (%d/%d 点 = %.1f%%) ==="
          % (len(changed), len(rows), 100.0 * len(changed) / len(rows)))
    for s in spans:
        print("  目標 %.3f--%.3f: %s (val %.4f, %.2f ms) -> %s (val %.4f, %.2f ms)"
              % (s["from"], s["to"], fmt(s["before"]), s["before"]["val"], s["before"]["time_ms"],
                 fmt(s["after"]), s["after"]["val"], s["after"]["time_ms"]))
    if not spans:
        print("  (変化なし)")
    out["b_target_sweep"] = {
        "note": ("目標 %.3f-%.3f を %.3f 刻みで振り、deployed の val 平均で目標を満たす構成のうち "
                 "%s が最短のものを選ぶ (同値なら低い解像度)。before = 既存候補のみ、"
                 "after = ViT-S/16 FP32 の 14 構成を追加" % (args.lo, args.hi, args.step, TKEY)),
        "n_target": len(rows), "n_changed": len(changed),
        "changed_fraction": len(changed) / len(rows),
        "spans": spans, "rows": rows}

    # ---- (c) 時間予算ごとの最良構成 ----
    print("\n=== (c) 時間予算ごとの最良構成 (deployed 選択) ===")
    bud = {}
    for T in O.TIME_BUDGETS:
        row = {}
        for tkey in ("gpu_ms", "e2e_ms", "e2e_deploy_ms"):
            a = pick_budget(base, T, tkey)
            b = pick_budget(both, T, tkey)
            ch = bool(a and b and (a["model"], a["res"]) != (b["model"], b["res"]))
            row[tkey] = {"before": budget_row(a) if a else None,
                         "after": budget_row(b) if b else None, "changed": ch}
            if tkey == TKEY:
                print("  %6.1f ms | %-22s val %.4f (%6.2f ms) -> %-22s val %.4f (%6.2f ms) %s"
                      % (T, fmt(a), a["acc_sel"], a[tkey], fmt(b), b["acc_sel"], b[tkey],
                         "<<< 変わる" if ch else ""))
        bud["%.1f" % T] = row
    n_ch = sum(1 for v in bud.values() for k in v if v[k]["changed"])
    print("  変化したセル: %d / %d (予算 4 x 時間軸 3)" % (n_ch, 4 * 3))
    out["c_time_budget"] = {
        "note": ("予算 T 以内で val 平均が最大の構成 (同値なら短い方 -> 低い解像度)。"
                 "optimal_n.py:403-413 と同一規則"),
        "budgets_ms": O.TIME_BUDGETS, "n_changed_cells": n_ch, "by_budget": bud}

    # ---- (d) パレートフロント ----
    print("\n=== (d) パレートフロント (支配判定は val 平均) ===")
    par = {}
    for tkey in ("gpu_ms", "e2e_deploy_ms", "e2e_ms"):
        fr0 = O.pareto(base, tkey)
        fr1 = O.pareto(both, tkey)
        set0 = [(p["model"], p["res"]) for p in fr0]
        set1 = [(p["model"], p["res"]) for p in fr1]
        entered = [p for p in fr1 if p["model"] == FP32_MODEL]
        dropped = [p for p in fr0 if (p["model"], p["res"]) not in set1]
        # 落ちた点を、どの FP32 構成が支配したかへ割り当てる
        by_fp32 = {}
        for p in dropped:
            killers = [q for q in fps if dominates(q, p, tkey)]
            for q in killers:
                by_fp32.setdefault("%s_r%d" % (q["model"], q["res"]), []).append(
                    {"model": p["model"], "res": p["res"], "acc_sel": round(p["acc_sel"], 4),
                     tkey: p[tkey]})
        par[tkey] = {
            "n_before": len(fr0), "n_after": len(fr1),
            "fp32_entered": [{"res": p["res"], "acc_sel": round(p["acc_sel"], 4),
                              "acc": round(p["acc"], 4), tkey: p[tkey],
                              "gpu_ms": p["gpu_ms"], "e2e_deploy_ms": p["e2e_deploy_ms"],
                              "e2e_ms": p["e2e_ms"]} for p in entered],
            "dropped_from_front": [{"model": p["model"], "res": p["res"],
                                    "acc_sel": round(p["acc_sel"], 4), tkey: p[tkey]}
                                   for p in dropped],
            "dominated_by_fp32": by_fp32,
            "front_after": [{"model": p["model"], "res": p["res"],
                             "acc_sel": round(p["acc_sel"], 4), "acc": round(p["acc"], 4),
                             tkey: p[tkey]} for p in fr1],
        }
        print("  [%s] %d -> %d 点 / FP32 が入る構成: %s"
              % (tkey, len(fr0), len(fr1),
                 ", ".join("N=%d" % p["res"] for p in entered) or "なし"))
        for p in dropped:
            print("      押し出された: %s (val %.4f, %.2f ms)"
                  % (fmt(p), p["acc_sel"], p[tkey]))
    out["d_pareto"] = {"note": ("optimal_n.pareto() と同一の支配判定 (時間最小・acc_sel 最大)。"
                                "front_after は FP32 を加えたあとのフロント。"
                                "⚠️ dominated_by_fp32 は「その落ちた点を支配する FP32 構成」を"
                                "すべて挙げるので、**新フロントに入らなかった FP32 構成も現れる** "
                                "(例: e2e_ms の N=208 は自身が N=224 に支配されるがフロント外)"),
                       "by_time_axis": par}

    # ---- (e) 0.93 前後の比較 ----
    def get(pts, m, n):
        return next(p for p in pts if p["model"] == m and p["res"] == n)
    cmp_specs = [(FP32_MODEL, 224), ("dinov2_l", 64), ("dinov2_l", 144),
                 (FP32_BASE, 224), ("resnet50", 224)]
    print("\n=== (e) 0.93 前後の比較 ===")
    print("%-22s %5s | %8s %8s | %9s %9s %9s" %
          ("モデル", "N", "val", "test", "gpu[ms]", "配備e2e", "オフe2e"))
    cmp_rows = []
    for m, n in cmp_specs:
        p = get(both, m, n)
        cmp_rows.append({"model": m, "res": n, "input_res": p["input_res"],
                         "val": round(p["acc_sel"], 4), "test": round(p["acc"], 4),
                         "val_source": p["acc_sel_source"], "test_source": p["acc_source"],
                         "n_seeds": p["n_seeds"], "gpu_ms": p["gpu_ms"],
                         "e2e_deploy_ms": p["e2e_deploy_ms"], "e2e_ms": p["e2e_ms"]})
        print("%-22s %5d | %8.4f %8.4f | %9.3f %9.3f %9.3f" %
              (LABEL[m], n, p["acc_sel"], p["acc"], p["gpu_ms"],
               p["e2e_deploy_ms"], p["e2e_ms"]))
    out["e_around_093"] = {
        "note": ("0.93 帯の比較。val は選択に使う値・test は報告値。"
                 "ViT-S/16 (FP16) と ResNet50 の N=224 も参考に並べる"),
        "rows": cmp_rows}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("\n[saved] %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
