#!/usr/bin/env python3
"""精度目標を細かく振り、**サーバ FP32 の val で選ぶ場合と配備エンジンの val で選ぶ場合**の
最短構成を並べる (査読指摘 A-1b).

背景: 論文は「目標 0.90 と 0.93 の最短構成がサーバ精度では ViT-S/16、配備精度では ResNet50 に
入れ替わる」と書いていた。しかしサーバ側の根拠に **test** の値 (0.9018 / 0.9357) を使い、
配備側は **val** で選んでいたため、**演算精度だけでなく分割も同時に変えていた**。
両方 val にそろえると **0.90 も 0.93 も入れ替わらない** (どちらも ResNet50)。

そこで「どこで入れ替わるのか」を目標の全域について出す。都合のよい 1 点だけを示さない。

⚠️ ViT-L の配備 val は測定中 (`run_vitl_acc_chain_val.sh`)。無い間はサーバ FP32 の val へ
   落とし、**落とした構成に `*` を付けて明示する** (黙って落とさない)。

usage:
  python3 train/target_sweep.py [--src trt10-maxn] [--step 0.001] [--out results/target_sweep.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "train"))
import optimal_n as O                                            # noqa: E402

R = os.path.join(HERE, "results")


def val_of(src, m, r):
    d = src["models"].get(m, {}).get(str(r), {}).get("B", {})
    return d.get("mean")


def pick(pts, src, fallback, target, tkey):
    """目標 target を満たす構成のうち tkey が最短のもの。落とし先を使ったら fb=True"""
    ok = []
    for p in pts:
        v = val_of(src, p["model"], p["res"])
        fb = False
        if v is None:
            v = val_of(fallback, p["model"], p["res"])
            fb = True
        if v is not None and v >= target:
            ok.append((p, v, fb))
    if not ok:
        return None
    # 同値時は短い方 -> 低い解像度 (optimal_n.py と同じ規則)
    p, v, fb = min(ok, key=lambda x: (x[0][tkey], x[0]["res"]))
    return {"model": p["model"], "res": p["res"], "val": v, "fallback": fb,
            "time_ms": p[tkey], "test": p["acc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="trt10-maxn")
    ap.add_argument("--step", type=float, default=0.001)
    ap.add_argument("--lo", type=float, default=0.80)
    ap.add_argument("--hi", type=float, default=0.975)
    ap.add_argument("--tkey", default="e2e_deploy_ms",
                    help="時間軸。配備経路 (既定) かオフライン経路 e2e_ms")
    ap.add_argument("--out", default=os.path.join(R, "target_sweep.json"))
    args = ap.parse_args()

    O.SRC = args.src
    pts, _, has_dep = O.load(select_acc_path=os.path.join(R, "summary_deploy_val.json"),
                             report_acc_path=os.path.join(R, "summary_deploy.json"),
                             select_fallback_path=os.path.join(R, "summary_val.json"))
    sv = json.load(open(os.path.join(R, "summary_val.json")))          # サーバ FP32 の val
    dv = json.load(open(os.path.join(R, "summary_deploy_val.json")))   # 配備 FP16 の val

    n_dep_models = sorted({m for m in dv["models"]})
    rows, changes = [], []
    t = args.lo
    while t <= args.hi + 1e-9:
        a = pick(pts, sv, sv, t, args.tkey)          # サーバ val (落とし先も自分自身)
        b = pick(pts, dv, sv, t, args.tkey)          # 配備 val (無い構成はサーバ val へ)
        rec = {"target": round(t, 4), "server": a, "deploy": b,
               "same": bool(a and b and (a["model"], a["res"]) == (b["model"], b["res"]))}
        rows.append(rec)
        if a and b and not rec["same"]:
            changes.append(rec)
        t += args.step

    # 入れ替わりが続く区間をまとめる
    spans, cur = [], None
    for rec in rows:
        diff = bool(rec["server"] and rec["deploy"] and not rec["same"])
        if diff and cur is None:
            cur = {"from": rec["target"], "to": rec["target"],
                   "server": rec["server"], "deploy": rec["deploy"]}
        elif diff:
            same_pair = ((cur["server"]["model"], cur["server"]["res"]) ==
                         (rec["server"]["model"], rec["server"]["res"]) and
                         (cur["deploy"]["model"], cur["deploy"]["res"]) ==
                         (rec["deploy"]["model"], rec["deploy"]["res"]))
            if same_pair:
                cur["to"] = rec["target"]
            else:
                spans.append(cur)
                cur = {"from": rec["target"], "to": rec["target"],
                       "server": rec["server"], "deploy": rec["deploy"]}
        elif cur is not None:
            spans.append(cur); cur = None
    if cur is not None:
        spans.append(cur)

    out = {"note": "選択規則を val にそろえたうえで、サーバ FP32 と配備 FP16 の "
                   "どちらで選ぶかだけを変えたときの最短構成",
           "src": args.src, "tkey": args.tkey, "step": args.step,
           "deploy_val_models": n_dep_models,
           "n_target": len(rows), "n_changed": len(changes),
           "changed_fraction": len(changes) / len(rows),
           "spans": spans, "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)

    print("=== 目標精度ごとの最短構成 (時間軸 %s・%s) ===" % (args.tkey, args.src))
    print("配備 val を持つモデル: %s" % ", ".join(n_dep_models))
    print("%8s | %-26s | %-26s" % ("目標", "サーバ FP32 の val で選ぶ", "配備 FP16 の val で選ぶ"))
    for rec in rows:
        if abs(rec["target"] * 1000 % 5) > 1e-6 and rec["same"]:
            continue                                    # 表示は 0.005 刻み + 変化点だけ
        def f(x):
            if not x:
                return "(該当なし)"
            return "%s N=%d %.4f%s %.2fms" % (O.LABEL[x["model"]], x["res"], x["val"],
                                              "*" if x["fallback"] else "", x["time_ms"])
        print("%8.3f | %-26s | %-26s %s" % (rec["target"], f(rec["server"]), f(rec["deploy"]),
                                            "" if rec["same"] else "<<< 変わる"))
    print("\n入れ替わる区間 (%d 件・全 %d 点中 %d 点 = %.1f%%):"
          % (len(spans), len(rows), len(changes), 100 * out["changed_fraction"]))
    for s in spans:
        print("  目標 %.3f--%.3f: サーバ %s N=%d (%.2f ms) -> 配備 %s N=%d (%.2f ms)"
              % (s["from"], s["to"], O.LABEL[s["server"]["model"]], s["server"]["res"],
                 s["server"]["time_ms"], O.LABEL[s["deploy"]["model"]], s["deploy"]["res"],
                 s["deploy"]["time_ms"]))
    print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
