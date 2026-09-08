#!/usr/bin/env python3
"""査読指摘 A1 で数値がどう変わるかを、新旧の optimal_n 出力から並べて出す.

サーバ FP32 の精度で作った表と、Orin の配備 FP16 の精度で作り直した表を突き合わせ、
**論文のどの記述を書き換えねばならないか**を一覧にする。

見るもの:
  表 5 (fastest_meeting_accuracy)   精度目標ごとの最短構成。A1 の主戦場
  パレートフロント (pareto_*)        構成の入れ替わり
  時間制約つき最良 (best_under_time_budget)
  飽和点 (saturation_point)

usage:
  python3 train/diff_a1.py --old results/optimal_n_trt10_maxn.json \
                           --new results/optimal_n_a1.json
"""
import argparse
import json
import sys

LABEL = {"mnv4": "MobileNetV4", "effb0": "EfficientNet-B0", "resnet50": "ResNet50",
         "vit_small": "ViT-S/16", "dinov2_l": "DINOv2-L", "dinov3_l": "DINOv3-L"}
TKEYS = ["gpu_ms", "e2e_ms", "e2e_deploy_ms"]
TNAME = {"gpu_ms": "GPU", "e2e_ms": "オフライン", "e2e_deploy_ms": "配備経路"}


def cfg(d):
    if not d:
        return "-"
    return "%s N=%d (%.4f, %.2f ms)" % (LABEL.get(d["model"], d["model"]), d["res"],
                                        d["acc"], d.get("e2e_deploy_ms", d.get("gpu_ms", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="サーバ FP32 で作った JSON")
    ap.add_argument("--new", required=True, help="配備精度で作り直した JSON")
    args = ap.parse_args()

    o = json.load(open(args.old))
    n = json.load(open(args.new))

    print("=== 出典 ===")
    print("  旧: %s  報告精度 %s" % (args.old, o.get("acc_report_source", "summary_v2.json")))
    print("  新: %s  報告精度 %s" % (args.new, n.get("acc_report_source", "?")))
    if n.get("acc_fallback_models"):
        print("      ⚠️ %s は配備精度が無くサーバ FP32 のまま"
              % ", ".join(n["acc_fallback_models"]))
    print("  seed 数  旧 %s / 新 %s" % (o.get("n_seeds"), n.get("n_seeds")))
    print("  構成数   旧 %s 点 / 新 %s 点" % (o.get("n_points"), n.get("n_points")))
    print("  選択元   旧 %s / 新 %s" % (o.get("acc_select_source", "(同一集合)"),
                                        n.get("acc_select_source", "(同一集合)")))

    # ⚠️ 測定途中で流すと、選択側 (validation) にまだ無い構成が候補から丸ごと落ちる。
    #    そのせいでパレートが激減し「配備精度にすると壊滅する」ように見えてしまうので、
    #    ここで必ず断っておく (2026-09-03 の試走では val 6 件の時点で 26 点 -> 5 点になった)。
    ns = n.get("n_seeds") or []
    thin = (n.get("n_points", 0) < o.get("n_points", 0)) or (ns and min(ns) < 29)
    if thin:
        print()
        print("  ⚠️⚠️ **新側の被覆が足りない。以下はすべて暫定値である。**")
        print("       構成数が減っているのは、選択側 (validation) にまだ CSV が無い構成が")
        print("       候補から外れるためであって、配備精度が低いからではない。")
        print("       fgpu0 の測定完了 (logs/deploy_acc_30seed_v2_all.done) 後に流し直すこと")

    # ---- 表 5 ----
    print("\n=== 表 5: 精度目標ごとの最短構成 (配備経路) ===")
    print("%-6s %-42s %-42s %s" % ("目標", "旧 (サーバ FP32)", "新 (配備 FP16)", "判定"))
    fo, fn = o.get("fastest_meeting_accuracy", {}), n.get("fastest_meeting_accuracy", {})
    changed = []
    for k in sorted(set(fo) | set(fn), key=float):
        a = (fo.get(k) or {}).get("e2e_deploy_ms")
        b = (fn.get(k) or {}).get("e2e_deploy_ms")
        if a and b and a["model"] == b["model"] and a["res"] == b["res"]:
            mark = "同じ"
        elif b is None:
            mark = "⛔ 達成不能へ"
            changed.append(k)
        elif a is None:
            mark = "新たに達成"
            changed.append(k)
        else:
            mark = "⭐ 入れ替わり"
            changed.append(k)
        print("%-6s %-42s %-42s %s" % (k, cfg(a), cfg(b), mark))
    if changed:
        print("  ⭐ 書き換えが要る目標: %s" % ", ".join(changed))
    else:
        print("  表 5 の構成は変わらない (数値だけ差し替えればよい)")

    # ---- パレート ----
    print("\n=== パレートフロントの構成数と入れ替わり ===")
    for tk in TKEYS:
        key = "pareto_" + tk
        po = {(p["model"], p["res"]) for p in o.get(key, [])}
        pn = {(p["model"], p["res"]) for p in n.get(key, [])}
        if not po and not pn:
            continue
        add = sorted(pn - po); rm = sorted(po - pn)
        print("  %-10s 旧 %2d 点 -> 新 %2d 点" % (TNAME.get(tk, tk), len(po), len(pn)))
        if add:
            print("      + %s" % ", ".join("%s N=%d" % (LABEL.get(m, m), r) for m, r in add))
        if rm:
            print("      - %s" % ", ".join("%s N=%d" % (LABEL.get(m, m), r) for m, r in rm))

    # ---- 時間制約つき最良 ----
    print("\n=== 時間制約つきの最良精度 (配備経路) ===")
    bo, bn = o.get("best_under_time_budget", {}), n.get("best_under_time_budget", {})
    for k in sorted(set(bo) | set(bn), key=float):
        a = (bo.get(k) or {}).get("e2e_deploy_ms")
        b = (bn.get(k) or {}).get("e2e_deploy_ms")
        same = a and b and a["model"] == b["model"] and a["res"] == b["res"]
        print("  %-6s ms  %-40s -> %-40s %s"
              % (k, cfg(a), cfg(b), "同じ" if same else "⭐ 入れ替わり"))

    # ---- 飽和点 ----
    print("\n=== モデルごとの飽和点 N* ===")
    so, sn = o.get("saturation_point", {}), n.get("saturation_point", {})
    for m in sorted(set(so) | set(sn)):
        a, b = so.get(m, {}), sn.get(m, {})
        na, nb = a.get("res"), b.get("res")
        aa, ab = a.get("acc"), b.get("acc")
        mark = "" if na == nb else "  ⭐"
        print("  %-16s N* %-5s -> %-5s   精度 %s -> %s%s"
              % (LABEL.get(m, m), na, nb,
                 "%.4f" % aa if aa else "-", "%.4f" % ab if ab else "-", mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
