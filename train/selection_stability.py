#!/usr/bin/env python3
"""**構成選択そのもの**が何シードで安定するかを測る (再レビュー指摘 (2)).

背景:
  seed_stability.py は「条件間の差の符号が 30 シードで安定するか」を見ているが、
  論文の主張は差の符号ではなく **「この目標／予算ならこの構成を選べ」** という表である。
  査読では「シード解析が構成選択の言葉に接続されていない」と指摘された。
  そこで本スクリプトは選択そのものを対象にする:

    「k シードの validation 平均だけで構成を選び直したとき、
      29 シード全部で選んだ構成とどれだけ一致するか。
      外したときに実際どれだけ損をするか (test 精度・配備時間)」

  一致率が k=1 で低く k が増えると 1 に近づくなら、「単一シードで選ぶのは危うい／
  何シードあれば選択が固まるか」を数値で言える。これは表の再現性そのものである。

⚠️ **共通 29 シード (43-71) に限定する。** CNN 4 種は seed 42-71 の 30 本あるが
   ViT-L (dinov2_l / dinov3_l) は 43-71 の 29 本しか測っていない (1 シード 126 パートの
   分割チェーンのため)。母数の違う平均どうしを比べると「シード数の差」が
   「モデルの差」に化けるので、**seed 42 を落として 29 本へ揃える**。

⚠️ **per_seed の並び順を推測しない。** collect_deploy_acc.py の実装 (下の
   `common_seeds()` のコメント参照) から根拠づけて対応させ、対応が保証できない
   構成があれば止まる。

選択規則は optimal_n.py (表 C / 表 D) と同一:
  精度目標 T : val 平均 >= T を満たすうち e2e_deploy_ms 最短 (同値なら低解像度)
  時間予算 B : e2e_deploy_ms <= B のうち val 平均最大 (同値なら短時間 -> 低解像度)

usage:
  python3 train/selection_stability.py [--subsets 2000] [--out results/selection_stability.json]
"""
import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import optimal_n as O  # noqa: E402

SELECT_ACC = os.path.join(ROOT, "results", "summary_deploy_val.json")
REPORT_ACC = os.path.join(ROOT, "results", "summary_deploy.json")
SELECT_FB = os.path.join(ROOT, "results", "summary_val.json")
OPTIMAL_N_REF = os.path.join(ROOT, "results", "optimal_n_a1.json")
SRC = "trt10-maxn"

RNG_SEED = 20260910
DROP_SEED = 42          # ViT-L に無いシード。落とす理由は上の注記のとおり

# 論文の表 (results/optimal_n_a1.json = 30/29 シード混在で選んだもの) の構成。
# 29 シードへ揃えたときに変わらないかを照合する。
PAPER_TARGET = {"0.80": ("resnet50", 32), "0.85": ("resnet50", 64),
                "0.90": ("resnet50", 112), "0.93": ("resnet50", 224),
                "0.95": ("dinov2_l", 144), "0.96": ("dinov2_l", 144)}
PAPER_BUDGET = {"10.0": ("dinov2_l", 64), "33.3": ("dinov3_l", 176),
                "50.0": ("dinov3_l", 208), "100.0": ("dinov3_l", 208)}


# ---------------------------------------------------------------- 入力の読み込み
def per_seed_matrix(sel):
    """summary_deploy_val.json から (構成, シード) の val 精度行列を作る.

    ⭐ per_seed の並びの根拠 (train/collect_deploy_acc.py を読んで確認した。推測ではない):

        mdirs = {m: dirs_for(m) for m in target_models}      # モデルごとの seed -> dir
        m_seeds = sorted(dirs)                               # 昇順のシード列
        for s in m_seeds:                                    # ★この順で
            ...
            accs.append(float((pred == labels).mean()))      # ★append される
        rec = {"per_seed": accs, ..., "n_seed": len(accs)}
        out["seeds_by_model"] = {m: sorted(mdirs[m]) for m in target_models}

      すなわち per_seed[i] は seeds_by_model[model][i] に対応する。
      ⚠️ ただし CSV が読めないシードは `continue` で**黙って飛ばされる**ので、
         飛ばしが起きた構成では対応が 1 つずつずれる。飛ばしが無いことは
         `n_seed == len(seeds_by_model[model])` で判定できるので、**必ず検査する**。
    """
    if "seeds_by_model" not in sel:
        sys.exit("[abort] summary_deploy_val.json に seeds_by_model が無い。"
                 "per_seed の並びを保証できないので中止する")
    sbm = sel["seeds_by_model"]
    table = {}
    for m, by_res in sel["models"].items():
        for r, d in by_res.items():
            b = d["B"]
            exp = sbm[m]
            if b["n_seed"] != len(exp):
                sys.exit("[abort] %s_r%s は n_seed=%d だが seeds_by_model は %d 本。"
                         "CSV の欠損で per_seed の並びがずれている可能性がある"
                         % (m, r, b["n_seed"], len(exp)))
            if len(b["per_seed"]) != len(exp):
                sys.exit("[abort] %s_r%s の per_seed 長 %d != %d"
                         % (m, r, len(b["per_seed"]), len(exp)))
            # 平均が一致することも確かめる (別物の列を掴んでいないかの保険)
            if abs(float(np.mean(b["per_seed"])) - b["mean"]) > 1e-9:
                sys.exit("[abort] %s_r%s の per_seed 平均が mean と食い違う" % (m, r))
            table[(m, int(r))] = dict(zip(exp, b["per_seed"]))
    return table, sbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=int, default=2000,
                    help="2<=k<=28 で引く部分集合の本数 (既定 2000)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "selection_stability.json"))
    args = ap.parse_args()

    sel = json.load(open(SELECT_ACC))
    rep = json.load(open(REPORT_ACC))
    per_seed, sbm = per_seed_matrix(sel)

    # ---- 共通シード -----------------------------------------------------------
    common = sorted(set.intersection(*[set(v) for v in sbm.values()]))
    dropped = sorted(set().union(*[set(v) for v in sbm.values()]) - set(common))
    print("=== 構成選択の安定性 (再レビュー指摘 (2)) ===")
    print("シード: モデル別 %s" % ", ".join("%s=%d" % (m, len(v)) for m, v in sbm.items()))
    print("  共通 %d 本 (%d-%d) / 除外 %s  ← 母数を揃えるため seed %s を落とす"
          % (len(common), common[0], common[-1], dropped, dropped))
    if dropped != [DROP_SEED]:
        print("  ⚠️ 除外シードが想定 (%d) と違う" % DROP_SEED)

    # ---- レイテンシと test 精度 ------------------------------------------------
    O.SRC = SRC
    pts, _, has_dep = O.load(select_acc_path=SELECT_ACC, report_acc_path=REPORT_ACC,
                             select_fallback_path=SELECT_FB)
    if not has_dep:
        sys.exit("[abort] e2e_deploy_ms が読めない (配備経路の前処理 JSON が無い)")
    if len({p["acc_sel_source"] for p in pts}) != 1 or pts[0]["acc_sel_source"] != "deployed_fp16":
        sys.exit("[abort] 選択精度がすべて配備実測ではない: %s"
                 % sorted({p["acc_sel_source"] for p in pts}))

    cfg = [(p["model"], p["res"]) for p in pts]
    time_ms = np.array([p["e2e_deploy_ms"] for p in pts])
    res_arr = np.array([p["res"] for p in pts])
    test_mean = np.array([p["acc"] for p in pts])          # summary_deploy.json の mean
    # 選択に使う (構成 x 共通シード) の val 精度
    V = np.array([[per_seed[c][s] for s in common] for c in cfg])
    n_cfg, n_seed = V.shape
    print("構成 %d 個 x 共通シード %d 本 / 時間 = e2e_deploy_ms (%s)" % (n_cfg, n_seed, SRC))

    # ---- 時間の値が optimal_n_a1.json と一致するか照合 --------------------------
    ref_json = json.load(open(OPTIMAL_N_REF))
    idx_of = {c: i for i, c in enumerate(cfg)}
    n_chk, bad = 0, []
    for sect, key in (("best_under_time_budget", "e2e_deploy_ms"),
                      ("fastest_meeting_accuracy", "e2e_deploy_ms")):
        for k, v in ref_json[sect].items():
            d = v[key]
            i = idx_of[(d["model"], d["res"])]
            n_chk += 1
            if abs(time_ms[i] - d["e2e_deploy_ms"]) > 1e-9:
                bad.append((sect, k, time_ms[i], d["e2e_deploy_ms"]))
    if bad:
        sys.exit("[abort] 時間が optimal_n_a1.json と食い違う: %s" % bad[:3])
    print("時間の照合: optimal_n_a1.json の %d セルすべてで e2e_deploy_ms が一致" % n_chk)

    # ---- 選択規則 (optimal_n.py と同一) ----------------------------------------
    # 時間昇順・同値なら解像度昇順に並べておくと、np.argmax(先頭の True/最大) が
    # そのまま「最短 (同値なら低解像度)」の同値規則になる。
    order = np.lexsort((res_arr, time_ms))
    V_o, time_o, res_o, test_o = V[order], time_ms[order], res_arr[order], test_mean[order]
    cfg_o = [cfg[i] for i in order]

    def sel_target(means, T):
        """means (n_cfg,) は order 済み. 返り値は order 済みの添字か None"""
        ok = means >= T
        return int(np.argmax(ok)) if ok.any() else None

    budget_cand = {B: np.where(time_o <= B)[0] for B in O.TIME_BUDGETS}

    def sel_budget(means, B):
        c = budget_cand[B]
        if c.size == 0:
            return None
        # argmax は最初の最大値を返す = 同値なら短時間 -> 低解像度
        return int(c[np.argmax(means[c])])

    # ---- 自己検査: モデル別の母数 (30/29) で選ぶと論文の表を再現するか -----------
    full_mean = np.array([np.mean(list(per_seed[c].values())) for c in cfg])[order]
    print("\n--- 自己検査: モデル別の全シード (CNN 30 / ViT-L 29) で選び直す ---")
    self_ok = True
    for T in O.ACC_TARGETS:
        i = sel_target(full_mean, T)
        got = cfg_o[i] if i is not None else None
        want = PAPER_TARGET["%.2f" % T]
        self_ok &= (got == want)
        print("  目標 %.2f: %-14s (論文 %-14s) %s" % (T, got, want, "OK" if got == want else "⛔"))
    for B in O.TIME_BUDGETS:
        i = sel_budget(full_mean, B)
        got = cfg_o[i] if i is not None else None
        want = PAPER_BUDGET["%.1f" % B]
        self_ok &= (got == want)
        print("  予算 %5.1f ms: %-14s (論文 %-14s) %s" % (B, got, want, "OK" if got == want else "⛔"))
    if not self_ok:
        sys.exit("[abort] 選択規則の実装が論文の表を再現しない。先にここを直すこと")
    print("  -> 実装は論文の表 (optimal_n_a1.json) を完全に再現する")

    # ---- 参照: 共通 29 シード全部の平均で選ぶ -----------------------------------
    ref_mean = V_o.mean(axis=1)
    print("\n--- 参照: 共通 %d シード (seed %d 除外) の平均で選び直す ---" % (n_seed, DROP_SEED))
    reference, ref_changed = {}, []
    for T in O.ACC_TARGETS:
        i = sel_target(ref_mean, T)
        key = "target_%.2f" % T
        want = PAPER_TARGET["%.2f" % T]
        got = cfg_o[i] if i is not None else None
        reference[key] = i
        same = (got == want)
        if not same:
            ref_changed.append((key, want, got))
        print("  目標 %.2f: %-14s (30/29 混在では %-14s) %s"
              % (T, got, want, "一致" if same else "⛔ 変化"))
    for B in O.TIME_BUDGETS:
        i = sel_budget(ref_mean, B)
        key = "budget_%.1f" % B
        want = PAPER_BUDGET["%.1f" % B]
        got = cfg_o[i] if i is not None else None
        reference[key] = i
        same = (got == want)
        if not same:
            ref_changed.append((key, want, got))
        print("  予算 %5.1f ms: %-14s (30/29 混在では %-14s) %s"
              % (B, got, want, "一致" if same else "⛔ 変化"))
    if ref_changed:
        print("  ⚠️⚠️ 29 シードへ揃えると %d 件の選択が変わる。これ自体が報告事項である"
              % len(ref_changed))
    else:
        print("  -> 29 シードへ揃えても論文の表と完全に一致する")

    # ---- k シード部分集合での選択 ----------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    criteria = ([("target_%.2f" % T, "target", T) for T in O.ACC_TARGETS] +
                [("budget_%.1f" % B, "budget", B) for B in O.TIME_BUDGETS])
    per_k = {}
    print("\n部分集合の選択を計算中 (k=1..%d) ..." % n_seed)
    for k in range(1, n_seed + 1):
        n_poss = math.comb(n_seed, k)
        if k == 1:
            subs = np.arange(n_seed).reshape(-1, 1)          # 全数 (29 通り)
            mode = "exhaustive"
        elif k == n_seed:
            subs = np.arange(n_seed).reshape(1, -1)          # 1 通り
            mode = "exhaustive"
        else:
            # 固定 rng で args.subsets 本。1 本の中は非復元 (同じシードを 2 度使わない)。
            subs = np.array([rng.choice(n_seed, size=k, replace=False)
                             for _ in range(args.subsets)])
            mode = "random"
        means = V_o[:, subs].mean(axis=2)                    # (n_cfg, n_sub)
        n_sub = subs.shape[0]
        rec = {"k": k, "n_subsets": int(n_sub), "mode": mode,
               "n_possible_subsets": int(n_poss),
               "n_distinct_subsets": int(len({tuple(sorted(s)) for s in subs.tolist()})),
               "by_criterion": {}}
        for name, kind, thr in criteria:
            ri = reference[name]
            picks = []
            for j in range(n_sub):
                mj = means[:, j]
                picks.append(sel_target(mj, thr) if kind == "target" else sel_budget(mj, thr))
            n_na = sum(1 for p in picks if p is None)
            n_hit = sum(1 for p in picks if p is not None and p == ri)
            mis = [p for p in picks if p is not None and p != ri]
            dacc = [(test_o[p] - test_o[ri]) * 100 for p in mis]
            dt = [time_o[p] - time_o[ri] for p in mis]
            cnt = Counter("%s_N%d" % cfg_o[p] if p is not None else "(該当なし)" for p in picks)
            rec["by_criterion"][name] = {
                "agree_rate": n_hit / n_sub,
                "n_agree": n_hit, "n_mismatch": len(mis), "n_none": n_na,
                "none_rate": n_na / n_sub,
                "mismatch_mean_abs_dtest_acc_pt": float(np.mean(np.abs(dacc))) if dacc else 0.0,
                "mismatch_mean_dtest_acc_pt": float(np.mean(dacc)) if dacc else 0.0,
                "mismatch_mean_abs_dtime_ms": float(np.mean(np.abs(dt))) if dt else 0.0,
                "mismatch_mean_dtime_ms": float(np.mean(dt)) if dt else 0.0,
                "top_selected": [{"config": c, "n": n, "rate": n / n_sub}
                                 for c, n in cnt.most_common(3)],
            }
        per_k[str(k)] = rec

    # ---- 出力 ------------------------------------------------------------------
    def cfgname(i):
        return "%s_N%d" % cfg_o[i] if i is not None else None

    out = {
        "note": ("k シードの validation 平均で構成選択をやり直したとき、共通 29 シード全部での"
                 "選択とどれだけ一致するか (再レビュー指摘 (2): シード解析を構成選択へ接続する)"),
        "rng_seed": RNG_SEED,
        "n_subsets_per_k": args.subsets,
        "subset_sampling": ("k=1 と k=29 は全数。2<=k<=28 は numpy default_rng(%d) で "
                            "%d 本を非復元抽出 (1 本の中で同じシードを 2 度使わない)。"
                            "C(29,k)<=%d の k では同じ部分集合が重複して引かれうるので "
                            "n_distinct_subsets を併記する" % (RNG_SEED, args.subsets, args.subsets)),
        "seeds_used": common,
        "n_seed": n_seed,
        "seeds_excluded": dropped,
        "seeds_excluded_reason": ("ViT-L (dinov2_l / dinov3_l) は seed 43-71 の 29 本しか"
                                  "配備精度を測っていないため、CNN の seed 42 を落として"
                                  "共通 29 本へ揃えた。母数の違う平均を比べると"
                                  "シード数の差がモデルの差に化ける"),
        "seeds_by_model": sbm,
        "select_acc": os.path.relpath(SELECT_ACC, ROOT),
        "report_acc": os.path.relpath(REPORT_ACC, ROOT),
        "latency_source": SRC,
        "latency_key": "e2e_deploy_ms",
        "latency_cross_checked_against": os.path.relpath(OPTIMAL_N_REF, ROOT),
        "n_latency_cells_checked": n_chk,
        "n_config": n_cfg,
        "selection_rule": {
            "target": "val 平均 >= T のうち e2e_deploy_ms 最短 (同値なら低解像度)",
            "budget": "e2e_deploy_ms <= B のうち val 平均最大 (同値なら短時間 -> 低解像度)",
        },
        "reference_selection": {
            k: {"config": cfgname(i),
                "val_mean_29seed": float(ref_mean[i]) if i is not None else None,
                "test_mean": float(test_o[i]) if i is not None else None,
                "e2e_deploy_ms": float(time_o[i]) if i is not None else None}
            for k, i in reference.items()},
        "paper_table_selection": {**{"target_%s" % k: "%s_N%d" % v for k, v in PAPER_TARGET.items()},
                                  **{"budget_%s" % k: "%s_N%d" % v for k, v in PAPER_BUDGET.items()}},
        "reference_matches_paper_table": not ref_changed,
        "reference_changes_vs_paper_table": [
            {"criterion": k, "paper": "%s_N%d" % w, "ref_29seed": ("%s_N%d" % g) if g else None}
            for k, w, g in ref_changed],
        "by_k": per_k,
    }

    # ---- stdout 要約表 ----------------------------------------------------------
    print("\n=== 要約: 参照 (29 シード) の選択に一致する割合 ===")
    print("%-14s %-16s %8s %8s %8s %8s %10s" %
          ("基準", "参照の構成", "k=1", "k=5", "k=10", "k>=95%", "不一致時"))
    print("%-14s %-16s %8s %8s %8s %8s %10s" % ("", "", "", "", "", "最小 k", "|Δacc| pt"))
    summary_rows = []
    for name, _, _ in criteria:
        a = {int(k): v["by_criterion"][name]["agree_rate"] for k, v in per_k.items()}
        k95 = next((k for k in sorted(a) if all(a[j] >= 0.95 for j in sorted(a) if j >= k)), None)
        k95_first = next((k for k in sorted(a) if a[k] >= 0.95), None)
        d1 = per_k["1"]["by_criterion"][name]
        print("%-14s %-16s %7.1f%% %7.1f%% %7.1f%% %8s %9.2f" %
              (name, cfgname(reference[name]), 100 * a[1], 100 * a[5], 100 * a[10],
               k95 if k95 else "-", d1["mismatch_mean_abs_dtest_acc_pt"]))
        summary_rows.append({"criterion": name, "reference": cfgname(reference[name]),
                             "agree_k1": a[1], "agree_k5": a[5], "agree_k10": a[10],
                             "k_for_95pct_stable": k95, "k_for_95pct_first": k95_first,
                             "k1_mismatch_mean_abs_dtest_acc_pt":
                                 d1["mismatch_mean_abs_dtest_acc_pt"],
                             "k1_mismatch_mean_abs_dtime_ms": d1["mismatch_mean_abs_dtime_ms"],
                             "k1_none_rate": d1["none_rate"]})
    out["summary"] = summary_rows

    print("\n=== 精度目標で「目標を満たせない」部分集合の割合 (該当なし) ===")
    for name, kind, _ in criteria:
        if kind != "target":
            continue
        rates = [(int(k), v["by_criterion"][name]["none_rate"]) for k, v in per_k.items()]
        rates = [r for r in sorted(rates) if r[1] > 0]
        print("  %-14s %s" % (name, ", ".join("k=%d %.1f%%" % (k, 100 * r) for k, r in rates[:8])
                              or "なし"))

    print("\n=== 不一致時の代償 (k=1 / k=5) ===")
    print("%-14s %-24s %-24s" % ("基準", "k=1  |Δacc| pt / |Δt| ms", "k=5  |Δacc| pt / |Δt| ms"))
    for name, _, _ in criteria:
        a = per_k["1"]["by_criterion"][name]
        b = per_k["5"]["by_criterion"][name]
        print("%-14s %10.2f / %8.2f      %10.2f / %8.2f" %
              (name, a["mismatch_mean_abs_dtest_acc_pt"], a["mismatch_mean_abs_dtime_ms"],
               b["mismatch_mean_abs_dtest_acc_pt"], b["mismatch_mean_abs_dtime_ms"]))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
