#!/usr/bin/env python3
"""ViT-L の Orin 実測 (latency + 精度) を 1 つの JSON にまとめる.

Phase 3 では ViT-L が Orin でビルドできず `orin_vs_h200` の 2 モデルが null のままだった。
8/18-19 の追加実験で両モデルとも動くようになったので、その結果を集計表へ載せる。

注意: **ViT-L の Orin 実測は r32 だけ**である (分割エンジンを r32 で作ったため)。
既存の `orin_vs_h200` は N=224 の表なので、そこへ混ぜてはいけない。
H200 側も r32 の値と突き合わせる。

latency は trtexec が各パートを単独で測った GPU Compute Time の mean を合計する
(パート間はデバイスメモリを渡すだけでホストコピーは発生しない)。

⭐ データ源 (--src) — 2026-08-31 に追加
  JetPack 6.2 化で TensorRT が 8.5.2 -> 10.3.0 になり、**測定の置き場・命名・段構成が
  すべて変わった**。旧データの再現性を壊さずに新データも読めるよう、データ源を選べるようにした。

  legacy      JetPack 5.1.2 / TRT 8.5.2   logs/meas30_*.median, logs/meas_*.median, preds_orin/
  trt10-15w   JetPack 6.2   / TRT 10.3    results_trt10_vitl_15w/*.json,  preds_trt10/
  trt10-maxn  JetPack 6.2   / TRT 10.3    results_trt10_vitl_maxn/*.json, preds_trt10/

  ⚠️ **段構成が新旧で違うのは全段 FP32 対照だけ**である (_chain を見よ)。配備構成は
     新旧とも DINOv2-L 5 段 / DINOv3-L 4 段で変わらない。
  ⚠️ **ORT 経路の比較は TRT 10.3 では測っていない** (measure_trt10_vitl_mode.sh に無い)。
     trt10 では preparation_path_comparison_dinov2_l を出さない。論文の当該記述は
     旧環境の値のままになるので、差し替えのときに出典を書き分けること。

usage (HPC 上で実行):
    python3 collect_vitl_edge.py                    # 旧環境 (既定・出力名も従来どおり)
    python3 collect_vitl_edge.py --src trt10-maxn   # JP6.2 / TRT 10.3 / MAXN_SUPER
    python3 collect_vitl_edge.py --src trt10-15w    # JP6.2 / TRT 10.3 / 15W
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

# ⚠️ 環境変数は**検証用の差し替え口**である (本番では設定しない)。
#    collect_powermode.py の PM_ROOT と同じ発想で、実測データを汚さずに
#    新経路 (trt10) の分岐をサンドボックスで確かめるために置いている。
EDGE = Path(os.environ.get("VITL_EDGE_ROOT") or "/home1/gfsi/ufsi0002/bs2026-resolution-edge")
WORK = Path(os.environ.get("VITL_WORK_ROOT") or "/work/gfsi/ufsi0002/bs2026-resolution")

# ⭐ サーバ FP32 の参照先 (2026-09-01 に修正)
#
# ⛔ 2026-09-01 まで `results/T1_condB/preds/` (= models/ 由来) を読んでいたが、
#    **Orin のエンジンは models_s42 から書き出した ONNX で作られている**。
#    別チェックポイントと比べていたので、モデルの seed 差が「半精度化の影響」として
#    argmax 一致率に混入していた (DINOv3-L 全段 FP32 で 77.5-98.5% -> 実際は 99.9-100%)。
#
#    根拠 (一致率とは独立の証拠):
#      - docs/bs2026-resolution_報告書.md:867  export_onnx.py --models_dir models_s42
#      - train/prep_vitl_all.sh:31 / prep_vitl_res.sh:35  --models_dir "$PROJ/models_s42"
#      - models/best.pt と models_s42/best.pt は同サイズだが md5 が異なる別物
#      - CNN 側 (results/orin/results/fp16_accuracy.json) は元から s42 を参照していた
#
#    ⚠️ 旧値を再現したいときだけ base を指すこと (--srv_preds results/T1_condB/preds)。
SRV_PREDS = os.environ.get("VITL_SRV_PREDS") or "results/T1_condB_s42/preds"

# ---- データ源 (main() で --src により設定する。既定は旧環境で振る舞い不変) ----
SRC = "legacy"
# TRT 10.3 の測定 JSON の置き場 (measure_trt10_vitl_mode.sh の OUT)
TRT10_DIRS = {"trt10-15w": "results_trt10_vitl_15w",
              "trt10-maxn": "results_trt10_vitl_maxn"}
# TRT 10.3 の全数推論 CSV の置き場 (infer_trt10_all.sh:24 の OUTDIR。旧は preds_orin/)
TRT10_PREDS = "preds_trt10"
SRC_ENV = {"legacy": {"jetpack": "5.1.2", "l4t": "R35.4.1", "trt": "8.5.2", "power_mode": "15W"},
           "trt10-15w": {"jetpack": "6.2", "l4t": "R36.4.3", "trt": "10.3.0", "power_mode": "15W"},
           "trt10-maxn": {"jetpack": "6.2", "l4t": "R36.4.3", "trt": "10.3.0",
                          "power_mode": "MAXN_SUPER"}}


def is_trt10():
    return SRC in TRT10_DIRS

# 各パートの latency をどのログから取るか (最終構成)
#
# DINOv2-L: p3 を FP16 でビルドすると必ず定数エンジンになる。2 分割して切り分けた結果
#   前半 p3s0 (blocks 18-20) は FP16 で正常、後半 p3s1 (blocks 21-23 + norm + head) が壊れる。
#   そこで p3s1 だけ FP32 を強制した 5 段構成が最速かつ正しい (p3 全体を FP32 にすると 7.50 ms)。
# DINOv3-L: 4 パートすべてをパート単位の ORT 最適化版でビルドした構成が最速。
PARTS = {
    "dinov2_l": [("p0", "logs/split_dinov2_l_r32_p0.log"),
                 ("p1", "logs/split_dinov2_l_r32_p1.log"),
                 ("p2", "logs/split_dinov2_l_r32_p2.log"),
                 ("p3s0 (FP16)", "logs/build_p3s0.log"),
                 ("p3s1 (FP32)", "logs/build_p3s1_fp32.log")],
    # DINOv3-L は全段 FP32 が配備構成 (FP16 では活性値の外れ値が飽和する)。
    # N=32 だけ FP16 でも通る場合があるが解像度を変えると再現しないため採らない。
    "dinov3_l": [("p0 (FP32)", "logs/build_v3_r32_p0_fp32.log"),
                 ("p1 (FP32)", "logs/build_v3_r32_p1_fp32.log"),
                 ("p2 (FP32)", "logs/build_v3_r32_p2_fp32.log"),
                 ("p3 (FP32)", "logs/build_v3_r32_p3_fp32.log")],
}
# 8/19 に最初に「正しく動く」ことを確認した構成 (最終構成より遅い)。経緯として残す
FIRST_CORRECT = {
    "dinov2_l": [("p0", "logs/split_dinov2_l_r32_p0.log"),
                 ("p1", "logs/split_dinov2_l_r32_p1.log"),
                 ("p2", "logs/split_dinov2_l_r32_p2.log"),
                 ("p3_ort (FP32)", "logs/build_p3_ort.log")],
    "dinov3_l": [("p0", "logs/split_dinov3_l_r32_p0.log"),
                 ("p1", "logs/split_dinov3_l_r32_p1.log"),
                 ("p2_ort", "logs/build_v3_p2_ort.log"),
                 ("p3_ort", "logs/build_v3_p3_ort.log")],
}
CSV = {"dinov2_l": "dinov2_l_r32_split5fp32_FP16.csv",
       "dinov3_l": "dinov3_l_r32_fp32full.csv"}
# 修正前 (定数を返していたエンジン) の構成。速度だけを見て正常と誤認しないための記録。
# 壊れたパートは計算が消えるぶん速く出るので、段別の内訳を残しておく
BROKEN_PARTS = {
    "dinov2_l": [("p0", "logs/split_dinov2_l_r32_p0.log", False),
                 ("p1", "logs/split_dinov2_l_r32_p1.log", False),
                 ("p2", "logs/split_dinov2_l_r32_p2.log", False),
                 ("p3", "logs/split_dinov2_l_r32_p3.log", True)],
    "dinov3_l": [("p0", "logs/split_dinov3_l_r32_p0.log", False),
                 ("p1", "logs/split_dinov3_l_r32_p1.log", False),
                 ("p2", "logs/split_dinov3_l_r32_p2.log", True),
                 ("p3", "logs/split_dinov3_l_r32_p3.log", True)],
}


RESOLUTIONS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
# 分割チェーンのパート名 (配備構成)
CHAIN = {"dinov2_l": ["p0", "p1", "p2", "p3s0", "p3s1"],
         "dinov3_l": ["p0", "p1", "p2", "p3"]}
PATCH = {"dinov2_l": 14, "dinov3_l": 16}
# CLS のみ (DINOv2) / CLS + register 4 (DINOv3)
EXTRA_TOKENS = {"dinov2_l": 1, "dinov3_l": 5}


def mean_ms(log):
    """trtexec のログから GPU Compute Time の mean を取り出す (ログが無ければ None)"""
    p = Path(log)
    if not p.exists():
        # 呼び出し側の一部 (main の PARTS/FIRST_CORRECT/BROKEN_PARTS ループ) は
        # 存在確認をしていない。旧環境では全ログが揃っていたので実害は無かったが、
        # 欠けると例外で落ちるだけなので None を返して `if ms:` に委ねる
        return None
    txt = p.read_text(errors="ignore")
    v = re.findall(r"GPU Compute Time:.*?mean = ([0-9.]+)", txt)
    return float(v[-1]) if v else None


def input_res(model, res):
    """モデルへ実際に入る一辺の画素数 (patch 倍数へ丸めた値)。train_res.resolve_input_res と同じ"""
    p = PATCH[model]
    return res if res % p == 0 else max(p, p * round(res / p))


def _tag(model, res, path):
    """測定ログ・エンジンの命名タグ (経路と解像度で変わる。履歴的な事情による)

    dinov2_l: f32 = 全段 FP32 の対照 / deployed かつ N!=32 = 生の分割 (v2raw) /
              それ以外 (N=32 の配備構成、および ORT 経由) = 初期の命名 (v2)
    dinov3_l: 常に v3
    """
    if model != "dinov2_l":
        return "v3"
    if path == "f32":
        return "v2f32"
    if path == "deployed" and res != 32:
        return "v2raw"
    return "v2"


def _chain(model, res, path):
    """段の並び。**全段 FP32 対照だけは段数がデータ源で変わる**

    p3 を p3s0/p3s1 に割ったのは「p3s1 だけ FP32 にする」ためで、全段 FP32 では
    分ける動機がない。

    legacy : N=32 だけ 4 段、他の水準は 5 段。
             N=32 には p3 の 2 分割 ONNX がそもそも存在しないため。
             ⚠️ /home1 には 4 段系統 (`f32s4_p{0..3}`・測定ログ 40 件) も残っているが、
                論文が採ったのは 5 段系統 (`meas_v2f32_r*_p3s0/p3s1.median`) の方である。
    trt10  : **全水準 4 段**。build_trt10_vitl.sh:136-142 が p3 を割らずに作るため。
             ⚠️ ここを 5 段のまま当てると p3s0/p3s1 が見つからず、sweep_rec の
                `if ms: tot += ms` で **p3 段が丸ごと欠落した過小な合計が警告なしに出る**。

    配備構成は新旧とも DINOv2-L 5 段 / DINOv3-L 4 段で変わらない
    (build_trt10_vitl.sh:121 が p0 p1 p2 p3s0 p3s1 を作る)。
    """
    if model == "dinov2_l" and path == "f32" and (is_trt10() or res == 32):
        return ["p0", "p1", "p2", "p3"]
    return CHAIN[model]


def _trt10_tag(model, res, part, path):
    """TRT 10.3 の測定 JSON のタグ (measure_trt10_vitl_mode.sh:151-177 の命名)

    配備構成   : dinov2_l_r112_p3s1 / dinov3_l_r112_p2
    全段 FP32  : dinov2_l_r112_f32_p3
    ⚠️ 旧の履歴的タグ (v2raw / v2 / v2f32 / v3) とは無関係である。_tag() は使わない。
    """
    if path == "f32":
        return "%s_r%d_f32_%s" % (model, res, part)
    return "%s_r%d_%s" % (model, res, part)


def _trt10_part(model, res, part, path):
    """TRT 10.3 の測定 JSON から 1 パートの中央値を取る (30 回別プロセス起動の中央値)"""
    if path == "ort":
        # ORT 経路は TRT 10.3 では測っていない。呼ばれたら黙って欠損にせず None を返す
        return None, None
    j = EDGE / TRT10_DIRS[SRC] / ("%s.json" % _trt10_tag(model, res, part, path))
    if not j.exists():
        return None, None
    d = json.load(open(j))
    ms = d.get("median_ms")
    if ms is None:
        return None, None
    n = d.get("n_ok") or d.get("reps")
    return round(float(ms), 3), ("median_of_%d_runs" % n if n else "median")


def part_ms(model, res, part, path="deployed"):
    """1 パートの latency。3 回測定の中央値 (.median) を優先し、無ければビルド時の値を使う

    path="deployed" は配備構成。DINOv2-L の配備構成は **ORT 最適化を通さない生の分割**で、
    ORT 経由の分割より 2.5 倍速い (ORT でグラフ構造が変わり MYELIN 融合が効かなくなる)。
    path="ort"  を渡すと比較用に ORT 経由の値を返す。
    path="f32"  を渡すと DINOv2-L の **全段 FP32** 対照 (build_v2_fullfp32.sh) の値を返す。

    ⭐ --src が trt10-* のときは測定 JSON から読む (置き場も命名も旧とは別物)。
    """
    if is_trt10():
        return _trt10_part(model, res, part, path)
    tag = _tag(model, res, path)
    # **30 回測定 (measure_all30.sh) があればそちらを優先**。3 回では中央値しか言えないが、
    # 30 回あれば標準偏差・95%CI まで報告できる (同名 .stats に入っている)。
    med30 = EDGE / ("logs/meas30_%s_r%d_%s.median" % (tag, res, part))
    if med30.exists():
        v = [float(x) for x in med30.read_text().split()]
        if v:
            return round(float(np.median(v)), 3), "median_of_30_runs"
    med = EDGE / ("logs/meas_%s_r%d_%s.median" % (tag, res, part))
    if med.exists():
        v = [float(x) for x in med.read_text().split()]
        if v:
            # .median には 3 回測定の中央値を 1 行だけ書いている (元の 3 値は同名 .log に残る)
            return round(float(np.median(v)), 3), "median_of_3_runs"
    cands = ["logs/build_%s_r%d_%s.log" % (tag, res, part)]
    if model == "dinov3_l":
        cands.insert(0, "logs/build_v3_r%d_%s_fp32.log" % (res, part))
    if res == 32:   # N=32 だけ初期の命名を使っている
        cands += {"dinov2_l": ["logs/split_dinov2_l_r32_%s.log" % part,
                               "logs/build_%s.log" % part,
                               "logs/build_p3s1_fp32.log" if part == "p3s1" else ""],
                  "dinov3_l": ["logs/build_v3_r32_%s_fp32.log" % part]}[model]
    for c in cands:
        if c and (EDGE / c).exists():
            v = mean_ms(EDGE / c)
            if v:
                return v, "single_run"
    return None, None


def sweep_rec(model, res, path="deployed"):
    """1 解像度ぶんの実測レコード (段別 latency・合計・精度・argmax 一致率)"""
    parts, tot, src = {}, 0.0, set()
    missing = []
    for p in _chain(model, res, path):
        ms, how = part_ms(model, res, p, path)
        parts[p] = ms
        if ms:
            tot += ms
            src.add(how)
        else:
            missing.append(p)
    ir = input_res(model, res)
    rec = {"parts": parts, "latency_ms": round(tot, 3), "input_res": ir,
           "tokens": (ir // PATCH[model]) ** 2 + EXTRA_TOKENS[model],
           "latency_source": "/".join(sorted(src)) if src else None}
    if is_trt10():
        # ⚠️ 上の `if ms:` は欠損段を黙って飛ばすので、1 段欠けると**過小な合計**が
        #    警告なしに出る。壊れたエンジンは計算が消えるぶん速く見えるため、
        #    latency だけでは異常に気づけない。新データでは欠損を必ず明示する
        rec["missing_parts"] = missing
        rec["complete"] = not missing

    h = EDGE / ("results_h200/%s_r%d.json" % (model, res))
    if h.exists() and tot:
        srv = json.load(open(h))["gpu_compute_mean_ms_median_of_reps"]
        rec["h200_ms"] = srv
        rec["edge_server_ratio"] = round(tot / srv, 1)

    # 精度 (全テスト 1,882 枚)。旧命名の CSV も拾う
    if is_trt10():
        # infer_trt10_all.sh:126-138 の命名。旧の別名 (raw_full / fp32full / CSV[]) は無い
        if model == "dinov2_l" and path == "f32":
            cands = ["%s/dinov2_l_r%d_f32_full.csv" % (TRT10_PREDS, res)]
        else:
            cands = ["%s/%s_r%d_full.csv" % (TRT10_PREDS, model, res)]
    else:
        cands = ["preds_orin/%s_r%d_full.csv" % (model, res)]
        if model == "dinov2_l" and path == "f32":
            cands = ["preds_orin/dinov2_l_r%d_f32_full.csv" % res]
        elif model == "dinov2_l" and path == "deployed":
            cands.insert(0, "preds_orin/dinov2_l_r%d_raw_full.csv" % res)
        if model == "dinov3_l":
            cands.append("preds_orin/dinov3_l_r%d_fp32full.csv" % res)
        if res == 32:
            cands.append("preds_orin/%s" % CSV[model])
    for c in cands:
        csv = EDGE / c
        if not csv.exists():
            continue
        pred = read_csv(csv).argmax(1)
        ref = np.load(WORK / SRV_PREDS / ("%s_r%d.npz" % (model, res)))
        labels, srv = ref["labels"], ref["probs"].argmax(1)
        if len(pred) != len(labels):
            continue
        rec["n"] = int(len(pred))
        rec["orin_acc"] = round(float((pred == labels).mean()), 4)
        rec["server_fp32_acc"] = round(float((srv == labels).mean()), 4)
        rec["argmax_agreement"] = round(float((pred == srv).mean()), 4)
        rec["distinct_outputs"] = len({ln.split(",", 1)[1]
                                       for ln in open(csv).read().splitlines()[1:] if ln})
        break
    return rec


def read_csv(path):
    rows = []
    with open(path) as f:
        ncls = len(f.readline().strip().split(",")) - 2
        for line in f:
            v = line.strip().split(",")
            if len(v) == ncls + 2:
                rows.append([float(x) for x in v[1:1 + ncls]])
    return np.asarray(rows)


def main():
    global SRC, SRV_PREDS
    ap = argparse.ArgumentParser(description="ViT-L の Orin 実測を 1 つの JSON にまとめる")
    ap.add_argument("--src", default="legacy", choices=["legacy"] + sorted(TRT10_DIRS),
                    help="データ源 (既定 legacy = JetPack 5.1.2 / TRT 8.5.2)")
    ap.add_argument("--out", default=None,
                    help="出力先 JSON (既定はデータ源ごとの名前。legacy は従来どおり)")
    ap.add_argument("--srv_preds", default=None,
                    help="サーバ FP32 参照の置き場 (既定 results/T1_condB_s42/preds = "
                         "ONNX の書き出し元。旧値の再現には results/T1_condB/preds)")
    args = ap.parse_args()
    SRC = args.src
    if args.srv_preds:
        SRV_PREDS = args.srv_preds

    out = {"note": "ViT-L の Orin 実測は r32 のみ。N=224 の orin_vs_h200 とは混ぜない",
           "resolution": 32, "models": {},
           # ⚠️ どのチェックポイントと比べた一致率なのかを必ず残す
           "server_reference": SRV_PREDS}
    if is_trt10():
        # 旧環境の値と取り違えないよう、出典をファイルの先頭に残す
        out["source"] = SRC
        out["measurement_env"] = SRC_ENV[SRC]
        out["note"] += ("。**この JSON は %s (JetPack 6.2 / TRT 10.3) の測定**であり、"
                        "JetPack 5.1.2 / TRT 8.5.2 の旧値とは混ぜられない" % SRC)

    for model, parts in PARTS.items():
        rec = {"parts": {}, "latency_ms": None}
        total = 0.0
        if is_trt10():
            # 旧 PARTS はビルドログ由来 (JP5.1.2 の 1 回計測)。TRT 10.3 では
            # 対応するビルドログが無いので r32 の測定 JSON (30 回中央値) から作る
            for p in _chain(model, 32, "deployed"):
                ms, _how = part_ms(model, 32, p, "deployed")
                rec["parts"][p] = ms
                if ms:
                    total += ms
        else:
            for name, log in parts:
                ms = mean_ms(EDGE / log)
                rec["parts"][name] = ms
                if ms:
                    total += ms
        rec["latency_ms"] = round(total, 3)

        h = json.load(open(EDGE / ("results_h200/%s_r32.json" % model)))
        rec["h200_r32_ms"] = h["gpu_compute_mean_ms_median_of_reps"]
        rec["edge_server_ratio"] = round(rec["latency_ms"] / rec["h200_r32_ms"], 2)

        # 精度 (サーバ FP32 の予測 npz と Orin FP16 の実測 CSV)
        # 8/19 に最初に正しく動いた構成 (経緯として残す)
        fc, ftotal = {}, 0.0
        for name, log in FIRST_CORRECT[model]:
            ms = mean_ms(EDGE / log)
            fc[name] = ms
            if ms:
                ftotal += ms
        rec["first_correct_build"] = {"parts": fc, "latency_ms": round(ftotal, 3)}

        ref = np.load(WORK / SRV_PREDS / ("%s_r32.npz" % model))
        labels, srv = ref["labels"], ref["probs"].argmax(1)
        rec["server_fp32_acc"] = round(float((srv == labels).mean()), 4)
        # ⚠️ TRT 10.3 では全数推論をやり直しているので CSV の置き場も命名も別
        csv = (EDGE / ("%s/%s_r32_full.csv" % (TRT10_PREDS, model))) if is_trt10() \
            else (EDGE / ("preds_orin/%s" % CSV[model]))
        if csv.exists():
            pred = read_csv(csv).argmax(1)
            rec["orin_fp16_acc"] = round(float((pred == labels).mean()), 4)
            rec["argmax_agreement"] = round(float((pred == srv).mean()), 4)
            rec["n"] = int(len(pred))

        # 修正前の (無効な) 測定値。どの段が定数を返していたかも残す
        bp, btotal, bad = {}, 0.0, []
        for name, log, broken in BROKEN_PARTS[model]:
            ms = mean_ms(EDGE / log)
            bp[name] = ms
            if ms:
                btotal += ms
            if broken:
                bad.append(name)
        rec["broken_build"] = {"parts": bp, "latency_ms": round(btotal, 3),
                               "constant_output_parts": bad}
        if is_trt10():
            # この 2 つは 2026-08 の切り分けの経緯で、対応する測定が TRT 10.3 には無い。
            # 値は旧環境のビルドログのままなので、出典を明記して取り違えを防ぐ
            for k in ("first_correct_build", "broken_build"):
                rec[k]["source_env"] = "legacy (JetPack 5.1.2 / TRT 8.5.2)"
        out["models"][model] = rec

    # DINOv3-L は単体エンジンもある (旧環境の値。TRT 10.3 の単体は後段で別キーに入れる)
    ms = mean_ms(EDGE / "logs/build_dinov3_l_r32_fp16_sim.log") if \
        (EDGE / "logs/build_dinov3_l_r32_fp16_sim.log").exists() else 46.380
    out["models"]["dinov3_l"]["single_engine_ms"] = ms
    if is_trt10():
        out["models"]["dinov3_l"]["single_engine_ms_source_env"] = \
            "legacy (JetPack 5.1.2 / TRT 8.5.2)"

    # ---- 解像度スイープ (全 14 水準) ----
    # 構成は両モデルとも N=32 で確立した配備構成をそのまま適用する。
    #   DINOv2-L: 5 段 (p3s1 のみ FP32)、DINOv3-L: 4 段すべて FP32
    # latency は measure_vitl_all.sh の 3 回中央値 (.median) を優先し、なければビルド時の値。
    sweep = {str(r): sweep_rec("dinov2_l", r) for r in RESOLUTIONS}
    base = sweep["32"]["latency_ms"]
    for v in sweep.values():
        v["ratio_to_r32"] = round(v["latency_ms"] / base, 2) if base else None
    out["resolution_sweep_dinov2_l"] = {
        "note": "5 段構成 (p3s1 のみ FP32)。解像度ごとに分割点は独立に選ばれる",
        "by_res": sweep,
        # base が 0 になるのは測定が 1 件も無いときだけ (legacy では起きない)
        "r224_over_r32": round(sweep["224"]["latency_ms"] / base, 2) if base else None,
    }

    # ---- 準備経路の比較 (DINOv2-L) ----
    # 同じ重み・同じ分割点・同じ精度構成でも、分割前に onnxruntime の基本最適化を通すと
    # TensorRT の融合が効かなくなり 2.5 倍遅くなる。配備構成は「通さない」方である。
    # 解像度の効果を測るときは経路を揃えないと桁を間違える (実際 N=32 だけ経路が違っていた)。
    if is_trt10():
        # ⚠️ ORT 経路は TRT 10.3 では測っていない (measure_trt10_vitl_mode.sh に該当なし)。
        #    空の by_res を出すと「比較したが差が無かった」と誤読されるので、
        #    測っていないことを明示する。論文の当該記述は legacy の値のままになる
        out["preparation_path_comparison_dinov2_l"] = {
            "note": ("TRT 10.3 では ORT 経路のエンジンを作っていないため比較を出さない。"
                     "論文の当該記述 (ORT 経由の分割は 2.5 倍遅い) は "
                     "JetPack 5.1.2 / TRT 8.5.2 の測定である"),
            "measured": False,
        }
    else:
        cmp_ = {}
        for r in RESOLUTIONS:
            if r == 32:
                continue   # N=32 は ORT 経由でビルドしていない (最初から生の分割だった)
            o = sweep_rec("dinov2_l", r, path="ort")
            d = sweep[str(r)]
            if o["latency_ms"] and d["latency_ms"]:
                cmp_[str(r)] = {"deployed_raw_split_ms": d["latency_ms"],
                                "ort_optimized_split_ms": o["latency_ms"],
                                "slowdown": round(o["latency_ms"] / d["latency_ms"], 2)}
        out["preparation_path_comparison_dinov2_l"] = {
            "note": ("分割前に onnxruntime の基本最適化を通すか否かの比較。"
                     "通すとグラフ構造が変わって TensorRT の融合が効かなくなり遅くなる。"
                     "配備構成は通さない方 (生の分割)。DINOv3-L は If を畳む必要があるので通す"),
            "by_res": cmp_,
        }

    # ---- DINOv3-L の解像度スイープ (全 14 水準) ----
    # DINOv3-L は残差ストリームが 1.55e5 に達し fp16 上限 65,504 を超えるため、
    # 4 段すべて FP32 でないと定数エンジンになる。全水準を同一 recipe (全段 FP32) で揃えた。
    v3 = {str(r): sweep_rec("dinov3_l", r) for r in RESOLUTIONS}
    b3 = v3["32"]["latency_ms"]
    for v in v3.values():
        v["ratio_to_r32"] = round(v["latency_ms"] / b3, 2) if b3 else None
    out["resolution_sweep_dinov3_l"] = {
        "note": ("4 段すべて FP32 (--precisionConstraints=obey --layerPrecisions=*:fp32)。"
                 "FP16 では残差ストリームが fp16 レンジを超えて飽和し定数出力になる"),
        "by_res": v3,
        "r224_over_r32": round(v3["224"]["latency_ms"] / b3, 2) if b3 else None,
        "fp16_overflow": {"dinov3_l_boundary_max_abs": 155118,
                          "dinov2_l_p3s1_internal_max_abs": 356409,
                          "dinov2_l_p3s0_internal_max_abs": 87,
                          "fp16_max": 65504},
    }

    # ---- DINOv2-L の全段 FP32 対照 (図 6 用、代表 6 水準) ----
    # 配備構成 (p3s1 のみ FP32) と重み・分割点は同一で、変えたのは演算精度だけ。
    # 一致率が上がれば低下は半精度化に、上がらなければ低解像度そのものに帰属する。
    f32 = {}
    for r in [16, 32, 64, 112, 128, 224]:
        rec = sweep_rec("dinov2_l", r, path="f32")
        if rec.get("argmax_agreement") is not None:
            f32[str(r)] = rec
    if f32:
        out["resolution_sweep_dinov2_l_fullfp32"] = {
            "note": ("DINOv2-L の全段 FP32 対照 (配備構成は p3s1 のみ FP32)。"
                     "重み・分割点・準備経路は配備構成と同一で演算精度だけが違う"
                     + ("。**TRT 10.3 の対照は p3 を割らない 4 段**であり、"
                        "旧環境で論文が採った 5 段 (p3s0/p3s1) とは段構成が違う"
                        if is_trt10() else "")),
            "by_res": f32,
        }
        if is_trt10():
            # 段数が新旧で違うので明示する (legacy 側は既存 JSON を変えないため足さない)
            out["resolution_sweep_dinov2_l_fullfp32"]["n_stages"] = \
                len(_chain("dinov2_l", 112, "f32"))
        print("\n  === dinov2_l 全段 FP32 対照 ===")
        print("    N   入力  Orin(ms)   精度    一致率   出力")
        for r in sorted(f32, key=int):
            v = f32[r]
            print("  %4s  %4d  %8.3f  %6s  %7s  %5s"
                  % (r, v["input_res"], v["latency_ms"], v.get("orin_acc", "-"),
                     v.get("argmax_agreement", "-"), v.get("distinct_outputs", "-")))

    # N=32 の代表値は 3 回測定の中央値 (スイープ側) に揃える。
    # models セクションの値はビルド時の 1 回計測なので参考として残す
    for m, key in (("dinov2_l", "resolution_sweep_dinov2_l"), ("dinov3_l", "resolution_sweep_dinov3_l")):
        r32 = out[key]["by_res"]["32"]
        out["models"][m]["latency_ms_single_run_build"] = out["models"][m]["latency_ms"]
        out["models"][m]["latency_ms"] = r32["latency_ms"]
        # H200 比は latency が 1 件も無いと sweep_rec が入れない (legacy では常に在る)
        if "edge_server_ratio" in r32:
            out["models"][m]["edge_server_ratio"] = r32["edge_server_ratio"]

    if is_trt10():
        # ⭐ 論文の貢献 4「分割しないとエッジに載らない」の前提を検証する。
        #    TRT 8.5.2 では ViT-L 全体が 1 個の foreign node に融合され、その一括コンパイルが
        #    メモリ頂点を作ってビルドできなかった。10.3 で融合の粒度が変わっていれば
        #    前提そのものが変わるので、単体エンジンの成否と速度を必ず記録する。
        #    ⛔ 2026-09-01: 当初は FP16 で作った単体を無条件に採用しており、DINOv2-L では
        #       **定数を返す壊れたエンジンの 8.049 ms を「単体は分割より 7% 速い」と報告**していた
        #       (実測: 1882 枚中 1881 枚が同一 logits・acc 0.1201)。DINOv2-L は FP16 で飽和する
        #       (分割版で p3s1 だけ FP32 にしているのはこのため)。
        #       ⭐ **壊れたエンジンは計算が消えるぶん速いので latency では気づけない。**
        #       そこで (a) FP32 版があればそれを優先し、(b) 全数推論 CSV の相異なる出力を
        #       必ず数えてから採用する。
        se = {}
        for m in ("dinov2_l", "dinov3_l"):
            cands = [("%s_r32_single_fp32" % m, "FP32 (--fp16 なし)"),
                     ("%s_r32_single" % m, "既定の精度でビルドしたもの")]
            r, rejected = None, []
            for tag, how in cands:
                j = EDGE / TRT10_DIRS[SRC] / ("%s.json" % tag)
                if not j.exists():
                    continue
                d = json.load(open(j))
                # ⚠️ 出力が定数のエンジンは採用しない (latency は速く見えるが無意味)
                csv_p = EDGE / TRT10_PREDS / ("%s.csv" % tag)
                distinct = n_rows = None
                if csv_p.exists():
                    with open(csv_p) as f:
                        lines = [ln for ln in f.read().splitlines()[1:] if ln]
                    n_rows = len(lines)
                    distinct = len({ln.split(",", 1)[1] for ln in lines})
                cand = {"built": True, "precision": how, "tag": tag,
                        "latency_ms": d.get("median_ms"), "engine_MB": d.get("engine_MB"),
                        "n_ok": d.get("n_ok"), "distinct_outputs": distinct,
                        "n_images": n_rows}
                # ⚠️ 「相異なる出力 <= 1」では甘い。DINOv2-L の FP16 単体は **3/1882** で、
                #    1881 枚が同一 logits (acc 0.1201) なのに素通りしていた。
                #    正常なエンジンは logits が連続値なので 1882/1882 になる。5% を境にする。
                floor = max(2, int(0.05 * (n_rows or 0)))
                if distinct is None:
                    cand["usable"] = False
                    cand["why"] = "全数推論 CSV が無いので出力の正しさを確認できない"
                elif distinct < floor:
                    cand["usable"] = False
                    cand["why"] = ("実質的な定数出力 (相異なる出力 %d < %d)。半精度の飽和で"
                                   "壊れており、計算が消えるぶん latency は速く見える"
                                   % (distinct, floor))
                else:
                    cand["usable"] = True
                if cand["usable"] and r is None:
                    r = cand
                else:
                    cand.setdefault("why", "より確実な %s を優先した" % cands[0][1])
                    rejected.append(cand)
            if r is None:
                se[m] = {"built": bool(rejected), "usable": False,
                         "note": ("単体エンジンの測定 JSON が無い (ビルド失敗か測定未到達)"
                                  if not rejected else
                                  "ビルドはできたが出力が使えない"),
                         "rejected": rejected}
                continue
            chain = out["resolution_sweep_%s" % m]["by_res"].get("32", {}).get("latency_ms")
            if r["latency_ms"] and chain:
                r["split_chain_ms"] = chain
                r["single_over_split"] = round(r["latency_ms"] / chain, 2)
            if rejected:
                r["rejected"] = rejected
            se[m] = r
        out["single_engine_trt10"] = {
            "note": ("TRT 10.3 で分割せずに単体エンジンをビルドできるかの検証 (N=32)。"
                     "TRT 8.5.2 ではビルド自体が通らなかった。"
                     "⭐ **採用は usable=true のものだけ** (定数出力のエンジンは latency が"
                     "速く見えるので必ず全数推論の相異なる出力で弾く)。"
                     "built が true なら「分割しないと載らない」という前提は成り立たないが、"
                     "single_over_split が 1 を上回るなら分割は「精度を保ったまま FP16 を"
                     "使うための混合精度の粒度」として速度上も有利である"),
            "by_model": se,
        }

    if args.out:
        dst = Path(args.out)
    elif is_trt10():
        dst = WORK / ("results/orin_vitl/vitl_edge_summary_%s.json" % SRC.replace("-", "_"))
    else:
        dst = WORK / "results/orin_vitl/vitl_edge_summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print("[saved] %s" % dst)
    if is_trt10():
        e = SRC_ENV[SRC]
        print("  データ源 %s (JetPack %s / L4T %s / TRT %s / %s)"
              % (SRC, e["jetpack"], e["l4t"], e["trt"], e["power_mode"]))
        se = out.get("single_engine_trt10", {}).get("by_model", {})
        for m, v in se.items():
            print("  単体エンジン %-9s built=%s usable=%s %s %s"
                  % (m, v.get("built"), v.get("usable"), v.get("precision", ""),
                     ("%.3f ms (分割 %.3f ms の %.2f 倍) 相異なる出力 %s"
                      % (v["latency_ms"], v["split_chain_ms"], v["single_over_split"],
                         v.get("distinct_outputs")))
                     if v.get("single_over_split") else ""))
            # ⚠️ 弾いたものも必ず見せる (黙って捨てると「無かった」ことになる)
            for x in v.get("rejected", []):
                print("      ⚠️ 不採用 %-24s %s (%s ms・相異なる出力 %s)"
                      % (x.get("tag"), x.get("why"), x.get("latency_ms"),
                         x.get("distinct_outputs")))
        # ⚠️ 欠損段の総覧。1 段欠けた合計は「速くなった」ように見えるので必ず出す
        bad = []
        for key in ("resolution_sweep_dinov2_l", "resolution_sweep_dinov3_l",
                    "resolution_sweep_dinov2_l_fullfp32"):
            for r_, v in sorted(out.get(key, {}).get("by_res", {}).items(),
                                key=lambda kv: int(kv[0])):
                if v.get("missing_parts"):
                    bad.append("%s r%s: %s" % (key.replace("resolution_sweep_", ""),
                                               r_, ",".join(v["missing_parts"])))
        if bad:
            print("  ⚠️ 欠損段あり %d 構成 — その構成の合計は過小である:" % len(bad))
            for b in bad[:15]:
                print("       %s" % b)
            if len(bad) > 15:
                print("       ... 他 %d 件" % (len(bad) - 15))
        else:
            print("  欠損段なし (全構成で全段そろっている)")
    for m, r in out["models"].items():
        print("  %-9s %7.3f ms (H200 r32 %.3f ms, %.1f 倍) acc %.4f -> %.4f 一致 %.2f%%"
              % (m, r["latency_ms"], r["h200_r32_ms"], r["edge_server_ratio"],
                 r["server_fp32_acc"], r.get("orin_fp16_acc", float("nan")),
                 r.get("argmax_agreement", float("nan")) * 100))

    for m, key in (("dinov2_l", "resolution_sweep_dinov2_l"), ("dinov3_l", "resolution_sweep_dinov3_l")):
        print("\n  === %s の解像度スイープ ===" % m)
        print("    N   入力  Orin(ms)  H200(ms)  比    精度    一致率   出力  測定")
        for r in RESOLUTIONS:
            v = out[key]["by_res"][str(r)]
            print("  %4d  %4d  %8.3f  %8.3f  %5s  %6s  %7s  %5s  %s"
                  % (r, v["input_res"], v["latency_ms"], v.get("h200_ms", float("nan")),
                     v.get("edge_server_ratio", "-"), v.get("orin_acc", "-"),
                     v.get("argmax_agreement", "-"), v.get("distinct_outputs", "-"),
                     v.get("latency_source") or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
