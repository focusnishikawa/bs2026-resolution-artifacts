#!/usr/bin/env python3
"""ViT-S/16 を **Orin で FP32 のまま配備した**ときの精度と latency を 1 本へ集約する.

背景:
  査読指摘 A1 への対応で報告値を「配備するエンジンそのものの精度」へ差し替えた結果、
  ViT-S/16 は FP16 化で N>=96 の帯が 3-6 ポイント落ちることが分かり、配備候補から
  実質的に外れた (target_sweep の deployed 側で ViT-S は 0.92-0.93 の狭い帯にしか出ない)。
  そこで「**同じ ViT-S/16 を FP32 のまま配備したら候補に戻るのか**」を測った。
  FP32 は精度が落ちない代わりに GPU 時間が約 2.5 倍になる (N=224: 2.359 -> 5.920 ms)。
  本スクリプトはその実測 CSV / latency JSON を集計するだけで、判断は
  train/vits_fp32_candidacy.py が行う。

入力:
  results/preds_30seed_fp32/s<seed>/vit_small_r<N>.csv       test 予測 (1,882 行 + ヘッダ)
  results/preds_30seed_fp32_val/s<seed>/vit_small_r<N>.csv   val  予測 (1,876 行 + ヘッダ)
  results/preds_30seed/labels.npy                            test 正解 (1,882)
  results/preds_30seed_val/labels.npy                        val  正解 (1,876)
  results/latency_fp32/results_trt10_fp32_maxn/vit_small_r<N>.json   MAXN_SUPER
  results/latency_fp32/results_trt10_fp32_15w/vit_small_r<N>.json    15W
  seed 42-71 の 30 本 x 解像度 16..224 (16 刻み) 14 水準 = 各 split 420 本

出力:
  results/summary_deploy_fp32.json
    splits.<val|test>.vit_small_fp32.<N>.B.{per_seed, mean, std, macro_recall_mean, n_seed}
      -> results/summary_deploy_val.json / summary_deploy.json の models サブツリーと同形なので
         optimal_n.py の --select-acc / --report-acc へそのまま差し込める形に組み替えられる
    latency_fp32.<maxn|15w>.<N>.{median_ms, mean_ms, sd_ms, cv_pct, n_ok, reps, engine_MB}

集計の流儀 (既存と一致させるため train/collect_deploy_acc.py の関数をそのまま使う):
  read_argmax()  行数がラベル数と一致しない CSV は使わない (黙って切り捨てない)
  mean()/std()   std は不偏 (n-1)
  macro_recall() クラスごとの recall の単純平均 (6 クラス)
  per_seed はシード昇順。丸めない (既存 JSON も生値を持つ)。

usage:
  python3 train/aggregate_fp32.py [--out results/summary_deploy_fp32.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import collect_deploy_acc as C                                     # noqa: E402

R = os.path.join(ROOT, "results")
RES = C.RES                                     # [16, 32, ..., 224]
SEEDS = list(range(42, 72))                     # 30 本
MODEL_KEY = "vit_small_fp32"                    # 既存 4+2 モデルと衝突させない名前
CSV_MODEL = "vit_small"                         # CSV のファイル名側は素の model 名
N_CLASSES = 6

SPLITS = {
    "test": {"preds": os.path.join(R, "preds_30seed_fp32"),
             "labels": os.path.join(R, "preds_30seed", "labels.npy"),
             "n_rows_with_header": 1883},
    "val": {"preds": os.path.join(R, "preds_30seed_fp32_val"),
            "labels": os.path.join(R, "preds_30seed_val", "labels.npy"),
            "n_rows_with_header": 1877},
}
LAT = {"maxn": os.path.join(R, "latency_fp32", "results_trt10_fp32_maxn"),
       "15w": os.path.join(R, "latency_fp32", "results_trt10_fp32_15w")}

# ---- 検証済みスポット値 (これと外れたら集計がおかしい。黙って通さない) ----
SPOT_ACC = {("val", 112): 0.8997, ("val", 224): 0.9386,
            ("test", 112): 0.9015, ("test", 224): 0.9359}
SPOT_LAT = {("maxn", 112): 2.589, ("maxn", 224): 5.920}


def load_split(name, cfg):
    """1 split 分 (14 解像度 x 30 シード) を集計する"""
    if not os.path.exists(cfg["labels"]):
        sys.exit("[abort] labels が無い: %s" % cfg["labels"])
    labels = np.load(cfg["labels"])
    n = int(len(labels))
    out, missing = {}, []
    for r in RES:
        accs, mrs, n_const = [], [], 0
        for s in SEEDS:
            p = os.path.join(cfg["preds"], "s%d" % s, "%s_r%d.csv" % (CSV_MODEL, r))
            # ⚠️ 行数は「ヘッダ + ラベル数」でなければならない。read_argmax はヘッダを
            #    除いた行数で判定するので、ファイル側の行数もここで直接確かめる。
            with open(p) as f:
                nl = sum(1 for _ in f)
            assert nl == cfg["n_rows_with_header"], \
                "%s の行数が %d (期待 %d)" % (p, nl, cfg["n_rows_with_header"])
            pred, why = C.read_argmax(p, n)
            assert pred is not None, "%s が読めない: %s" % (p, why)
            if len(np.unique(pred)) <= 1:
                n_const += 1
            accs.append(float((pred == labels).mean()))
            mrs.append(C.macro_recall(pred, labels, N_CLASSES))
        if len(accs) != len(SEEDS):
            missing.append((r, len(accs)))
        rec = {"per_seed": accs, "mean": C.mean(accs), "std": C.std(accs),
               "macro_recall_mean": C.mean(mrs), "n_seed": len(accs)}
        if n_const:
            rec["n_const"] = n_const
        out[str(r)] = {"B": rec}
    if missing:
        sys.exit("[abort] シードが足りない構成: %s" % missing)
    return out, labels, n


def load_latency(tag, d):
    out = {}
    for r in RES:
        p = os.path.join(d, "%s_r%d.json" % (CSV_MODEL, r))
        j = json.load(open(p))
        assert j["precision"] == "fp32", "%s が FP32 でない: %s" % (p, j["precision"])
        assert int(j["res"]) == r, "%s の res が %s" % (p, j["res"])
        out[str(r)] = {"median_ms": j["median_ms"], "mean_ms": j["mean_ms"],
                       "sd_ms": j.get("sd_ms"), "cv_pct": j.get("cv_pct"),
                       "n_ok": j.get("n_ok"), "reps": j.get("reps"),
                       "engine_MB": j.get("engine_MB"),
                       "power_mode": j.get("power_mode"), "trt": j.get("trt")}
        assert out[str(r)]["n_ok"] == 30, "%s の n_ok が %s" % (p, out[str(r)]["n_ok"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "summary_deploy_fp32.json"))
    args = ap.parse_args()

    splits, meta = {}, {}
    for name, cfg in SPLITS.items():
        acc, labels, n = load_split(name, cfg)
        splits[name] = {MODEL_KEY: acc}
        meta[name] = {"labels": os.path.abspath(cfg["labels"]), "n": n,
                      "preds": os.path.relpath(cfg["preds"], ROOT),
                      "n_csv": len(RES) * len(SEEDS)}
        print("=== %s (n=%d, %d シード x %d 解像度 = %d CSV) ===" %
              (name, n, len(SEEDS), len(RES), len(RES) * len(SEEDS)))
        print("  %s" % "  ".join("%d:%.4f" % (r, acc[str(r)]["B"]["mean"]) for r in RES))

    lat = {tag: load_latency(tag, d) for tag, d in LAT.items()}
    print("\n=== latency FP32 (median_ms) ===")
    for tag in ("maxn", "15w"):
        print("  %-5s %s" % (tag, "  ".join("%d:%.3f" % (r, lat[tag][str(r)]["median_ms"])
                                            for r in RES)))

    # ---- スポット値の焼き込み検証 ----
    for (sp, r), want in SPOT_ACC.items():
        got = round(splits[sp][MODEL_KEY][str(r)]["B"]["mean"], 4)
        assert got == want, "%s N=%d の mean が %.4f (期待 %.4f)" % (sp, r, got, want)
    for (tag, r), want in SPOT_LAT.items():
        got = round(lat[tag][str(r)]["median_ms"], 3)
        assert got == want, "%s N=%d の median_ms が %.3f (期待 %.3f)" % (tag, r, got, want)
    print("\n[assert] スポット値 %d 点 (精度) + %d 点 (latency) 一致"
          % (len(SPOT_ACC), len(SPOT_LAT)))

    out = {"metric": "acc", "precision": "orin_fp32_deployed",
           "note": ("Orin Nano で **FP32 のまま**ビルドしたエンジン (TRT 10.3 / JetPack 6.2) を "
                    "実機で全数推論した精度と、同じエンジンの GPU Compute Time。"
                    "対象は ViT-S/16 のみ・条件 B (解像度別ネイティブ学習)・seed 42-71 の 30 本。"
                    "既存の summary_deploy*.json は FP16 配備なので別ファイルに分ける。"
                    "モデル名を vit_small_fp32 としているのは FP16 版と同時に候補へ並べるため"),
           "model_key": MODEL_KEY, "csv_model": CSV_MODEL,
           "seeds": SEEDS, "resolutions": RES,
           "splits": splits, "splits_meta": meta,
           "latency_fp32": lat,
           "latency_source": {tag: os.path.relpath(d, ROOT) for tag, d in LAT.items()},
           "latency_note": ("trtexec の GPU Compute Time。統計量は analyze_all.py の "
                            "load_latency(d, \"median_ms\") と同じ流儀で median を点値に使う"),
           "verified_spot_values": {"acc": {"%s_N%d" % k: v for k, v in SPOT_ACC.items()},
                                    "latency_median_ms": {"%s_N%d" % k: v
                                                          for k, v in SPOT_LAT.items()}}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("[saved] %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
