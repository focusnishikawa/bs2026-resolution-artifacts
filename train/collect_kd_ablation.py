#!/usr/bin/env python3
"""蒸留 ablation の結果を 1 つの JSON にまとめる.

PLAN.md の宿題: 既存 raptor の蒸留済み生徒 (effb0) が本 PJ の CE-only より低解像度で
大きく強かった (N=32 で 0.636 対 0.538) ため、「蒸留が低解像度耐性を高める」可能性が
指摘されていた。データ・split・レシピ・学習上限・3 seed を揃え、**損失だけを変えて**
検証した結果をここに集約する。

出力: results/kd_ablation/kd_ablation_summary.json
"""
import json
import statistics as st
from pathlib import Path

P = Path("/work/gfsi/ufsi0002/bs2026-resolution")
RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
SEEDS = [42, 43, 44]


def main():
    ce = json.load(open(P / "results/summary_v2.json"))["models"]["effb0"]
    out = {
        "question": "蒸留は低解像度耐性を高めるか",
        "design": {
            "student": "effb0 @ N=224 (条件 B の N=224 と同一設定)",
            "teacher": "dinov2_l @ N=224 (本 PJ Phase 1, test acc 0.9772)",
            "loss": "0.5 * T^2 * KL(student/T || teacher/T) + 0.5 * CE, T=2",
            "control": "データ・split・レシピ・学習上限・3 seed を CE-only と同一にし、違いは損失のみ",
            "evaluation": "条件 A (224 学習モデルへ N へ落とした入力) を 14 水準",
            "seeds": SEEDS,
        },
        "kd_train": {}, "per_resolution": {},
    }

    for s in SEEDS:
        m = json.load(open(P / ("models_kd/effb0_r224_kd_s%d/test_metrics.json" % s)))
        out["kd_train"][str(s)] = {k: m[k] for k in
                                   ("test_acc", "macro_recall", "best_epoch", "epochs_run", "train_sec")}

    n_pos = n_neg = 0
    for r in RES:
        kd = [json.load(open(P / ("results/T1_condA_kd_s%d/effb0_r%d.json" % (s, r))))["acc"]
              for s in SEEDS]
        cev = ce[str(r)]["A"]["per_seed"]
        dif = [(k - c) * 100 for k, c in zip(kd, cev)]
        sign = "+3/3" if all(x > 0 for x in dif) else ("-3/3" if all(x < 0 for x in dif) else "混在")
        n_pos += sign == "+3/3"
        n_neg += sign == "-3/3"
        out["per_resolution"][str(r)] = {
            "kd_per_seed": kd, "kd_mean": st.mean(kd), "kd_std": st.pstdev(kd),
            "ce_per_seed": cev, "ce_mean": ce[str(r)]["A"]["mean"], "ce_std": ce[str(r)]["A"]["std"],
            "diff_pt_per_seed": dif, "diff_pt_mean": st.mean(dif), "sign_agreement": sign,
        }

    lo = [out["per_resolution"][str(r)]["diff_pt_mean"] for r in RES if r <= 96]
    hi = [out["per_resolution"][str(r)]["diff_pt_mean"] for r in RES if r >= 112]
    out["conclusion"] = {
        "sign_agree_positive": n_pos, "sign_agree_negative": n_neg,
        "sign_mixed": len(RES) - n_pos - n_neg,
        "mean_diff_low_res_N<=96_pt": round(st.mean(lo), 3),
        "mean_diff_high_res_N>=112_pt": round(st.mean(hi), 3),
        "verdict": ("蒸留は低解像度耐性を高めない。N<=96 では 3 seed の符号が一つも揃わず "
                    "(平均 %.2f pt)、効果が見えるのは高解像度側 (N>=112 で平均 %.2f pt、"
                    "3 seed 符号一致は N=176 と N=224 のみ)。raptor で観測された "
                    "N=32 の +9.8 pt 差は蒸留では説明できない。" % (st.mean(lo), st.mean(hi))),
        "raptor_reference": {"effb0_kd_student": {"32": 0.636, "64": 0.807, "128": 0.905, "224": 0.928},
                             "note": "レシピ・データ・評価前処理が本 PJ と揃っていない別実験の値"},
    }

    dst = P / "results/kd_ablation/kd_ablation_summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print("[saved] %s" % dst)
    print(out["conclusion"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
