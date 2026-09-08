#!/usr/bin/env python3
"""Phase 2 v2 の集計: 3 seed の平均 +- 標準偏差で条件 A / 条件 B / 解像度適応の利得を出す.

出力:
  results/summary_v2.json  機械可読な全数値 (後段の作図・検定はここから読む)
  標準出力に表

利得の符号が seed に依存しないか (3 seed すべてで同符号か) を必ず確認する。
raptor 論文で「単一 seed 由来の差」を主張してしまった失敗を繰り返さないため。
"""
import argparse
import json
import glob
import os
import re

RES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
MODELS = ["mnv4", "effb0", "resnet50", "vit_small", "dinov2_l", "dinov3_l"]
SEEDS = [42, 43, 44]
PAT = re.compile(r"^(mnv4|effb0|resnet50|vit_small|dinov2_l|dinov3_l)_r(\d+)\.json$")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load(mode, seed, root):
    out = {}
    d = os.path.join(root, "results", "T1_cond%s_s%d" % (mode, seed))
    for f in glob.glob(os.path.join(d, "*.json")):
        if not PAT.match(os.path.basename(f)):
            continue
        j = json.load(open(f))
        out[(j["model"], j["res"])] = j
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/work/gfsi/ufsi0002/bs2026-resolution")
    p.add_argument("--metric", default="acc", choices=["acc", "macro_recall"])
    args = p.parse_args()

    # seed は実データから検出する。3 seed から 30 seed へ広げる途中は seed ごとに
    # 完成度が違うので、**条件 A/B とも 84 構成が揃った seed だけ**を平均に使う。
    # 中途半端な seed を混ぜると水準ごとに母数が変わり、比較そのものが壊れる。
    global SEEDS
    found = []
    for d in glob.glob(os.path.join(args.root, "results", "T1_condB_s*")):
        mm = re.search(r"_s(\d+)$", d)
        if not mm:
            continue
        s = int(mm.group(1))
        if len(load("A", s, args.root)) >= 84 and len(load("B", s, args.root)) >= 84:
            found.append(s)
    if found:
        found.sort()
        if found != SEEDS:
            print("[seeds] 完全な seed を検出: %d 本 %s" % (len(found), found))
        SEEDS = found
    else:
        print("[seeds] 完全な seed が無いので既定 %s を使う" % SEEDS)

    data = {m: {s: load(m, s, args.root) for s in SEEDS} for m in ["A", "B"]}
    n_have = {(m, s): len(data[m][s]) for m in ["A", "B"] for s in SEEDS}
    print("読み込み: " + ", ".join("cond%s_s%d=%d" % (m, s, n) for (m, s), n in sorted(n_have.items())))

    # ⚠️ 評価 JSON が 1 件も無いまま書き出すと summary_v2.json が nan で埋まり、
    #    図も後段の集計もすべて壊れる (実際に Mac で誤実行して上書きする事故を起こした)。
    #    生データは HPC 側にしか無いので、**本スクリプトは HPC で実行する**。
    if not sum(n_have.values()):
        print("[中断] 評価 JSON を 1 件も読めなかった。summary_v2.json は書き換えない。\n"
              "        本スクリプトは生データのある HPC 上で実行すること "
              "(/work/gfsi/ufsi0002/bs2026-resolution)。")
        return 1

    summary = {"metric": args.metric, "seeds": SEEDS, "resolutions": RES, "models": {}}

    for label, mode in [("条件A (劣化耐性)", "A"), ("条件B (解像度別ネイティブ学習)", "B")]:
        print("\n=== %s: %s  平均 (3 seed) ===" % (label, args.metric))
        print("%-10s" % "model" + "".join("%8d" % r for r in RES))
        for mo in MODELS:
            row = ""
            for r in RES:
                xs = [data[mode][s][(mo, r)][args.metric] for s in SEEDS if (mo, r) in data[mode][s]]
                row += ("%8.4f" % mean(xs)) if xs else "       -"
            print("%-10s" % mo + row)
        print("--- 標準偏差 ---")
        print("%-10s" % "model" + "".join("%8d" % r for r in RES))
        for mo in MODELS:
            row = ""
            for r in RES:
                xs = [data[mode][s][(mo, r)][args.metric] for s in SEEDS if (mo, r) in data[mode][s]]
                row += ("%8.4f" % std(xs)) if xs else "       -"
            print("%-10s" % mo + row)

    print("\n=== 解像度適応の利得: 条件B - 条件A (pt, 3 seed 平均) ===")
    print("%-10s" % "model" + "".join("%8d" % r for r in RES))
    for mo in MODELS:
        row = ""
        for r in RES:
            a = [data["A"][s][(mo, r)][args.metric] for s in SEEDS if (mo, r) in data["A"][s]]
            b = [data["B"][s][(mo, r)][args.metric] for s in SEEDS if (mo, r) in data["B"][s]]
            row += ("%+8.1f" % ((mean(b) - mean(a)) * 100)) if a and b else "       -"
        print("%-10s" % mo + row)

    print("\n=== 利得の符号一致 (3 seed すべてで同符号なら ○、混在なら x) ===")
    print("%-10s" % "model" + "".join("%8d" % r for r in RES))
    for mo in MODELS:
        row = ""
        for r in RES:
            ds = [data["B"][s][(mo, r)][args.metric] - data["A"][s][(mo, r)][args.metric]
                  for s in SEEDS if (mo, r) in data["A"][s] and (mo, r) in data["B"][s]]
            if len(ds) < len(SEEDS):
                row += "       -"
            else:
                row += "%8s" % ("○" if all(d > 0 for d in ds) or all(d < 0 for d in ds) else "x")
        print("%-10s" % mo + row)

    print("\n=== 所要画素数 (224 比 -5 pt 以内を保てる最小 N、3 seed 平均) ===")
    print("%-10s %10s %10s" % ("model", "条件A", "条件B"))
    for mo in MODELS:
        got = {}
        for mode in ["A", "B"]:
            base = mean([data[mode][s][(mo, 224)][args.metric] for s in SEEDS if (mo, 224) in data[mode][s]])
            ok = [r for r in RES
                  if all((mo, r) in data[mode][s] for s in SEEDS)
                  and (mean([data[mode][s][(mo, r)][args.metric] for s in SEEDS]) - base) * 100 >= -5.0]
            got[mode] = min(ok) if ok else None
        print("%-10s %10s %10s" % (mo, got["A"], got["B"]))

    for mo in MODELS:
        summary["models"][mo] = {}
        for r in RES:
            e = {}
            for mode in ["A", "B"]:
                xs = [data[mode][s][(mo, r)][args.metric] for s in SEEDS if (mo, r) in data[mode][s]]
                mrs = [data[mode][s][(mo, r)]["macro_recall"] for s in SEEDS if (mo, r) in data[mode][s]]
                e[mode] = {"per_seed": xs, "mean": mean(xs), "std": std(xs),
                           "macro_recall_mean": mean(mrs), "n_seed": len(xs)}
            summary["models"][mo][str(r)] = e
    out = os.path.join(args.root, "results", "summary_v2.json")
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    # 中断時に非 0 を返す (呼び出し側のシェルが失敗を検知できるようにする)
    raise SystemExit(main() or 0)
