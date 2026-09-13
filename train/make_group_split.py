#!/usr/bin/env python3
"""元画像・動画系列をまたがない group split を作る (査読指摘 A-3).

現行の分割は**画像単位**の層化ランダム分割であり、同一の元画像から切り出した別クロップが
train と test にまたがっている (test 18/1,882 枚 = 0.96%、validation 13/1,876 枚 = 0.69%)。
この枚数は選択表の目標との差 (数枚規模) と同程度なので、限界として書くだけでは足りない。

**group の定義** (rel_path から決める。誤ると漏洩が残るので種類ごとに明示する):

  1. `AAS<n>_<種>-<CAMID>-<x>-<y>-<w>-<h>.JPG` / `UDS_<種>-<CAMID>-...`
       -> group = 種 + CAMID。同じ元写真からの複数クロップが 1 group になる
  2. `frame_<n>_bb<n>_t<秒>_conf<p>.jpg`
       -> group = 種 + "VIDEO"。⚠️ 動画 ID がファイル名にもディレクトリにも無い。
          連番と時刻から同一動画の隣接フレームであることは明らかなので、
          **種ごとに全フレームを 1 group** とする (最も保守的)
  2'. `<動画ID8桁hex>__frame_<n>_bb<n>_t<秒>_conf<p>.jpg`  (--rule v2 のときだけ)
       -> group = 種 + "VIDEO" + 動画 ID
  3. `inat_<id>_<hash>.jpg`   -> group = "inat_<id>"   (同一観測の複数写真をまとめる)
  4. `ebird_<id>.jpg`         -> group = "ebird_<id>"
  5. それ以外 (BIRDS525/ や v5_ds_beta_bird/ の連番など)
       -> group = rel_path そのもの (1 枚 1 group。元画像を共有する根拠が無い)

分割は **group 単位の層化**で行う。クラスごとに group をシャッフルし、
枚数比が 7:1.5:1.5 に近づくよう大きい group から貪欲に詰める。

⭐ 群規則は 2 つある (--rule。既定 v1 = A-3 で実際に使った規則で完全に従来どおり)。

  v1 (既定)  上の 1-5。⚠️ 2 の RE_FRAME は接頭辞なし `frame_...jpg` しか拾わないので、
             `<動画ID8桁>__frame_...jpg` の 1,000 枚 (bird_frames_pure_oowashi_20260504/)
             が **1 枚 1 group (SINGLE)** になっていた。すなわち同一動画の隣接フレームが
             独立な group として train/val/test に散る。
  v2         2' を先頭に足して動画 ID ごとに 1 group へまとめる。
             規則は make_data_manifest.py:group_v2 と厳密に同一 (正規表現も同じ)。

usage:
  python3 train/make_group_split.py --report            # 現行分割の漏洩を数えるだけ
  python3 train/make_group_split.py --out data/splits_group
  python3 train/make_group_split.py --rule v2 --out data/splits_group_v2
"""
import argparse
import csv
import os
import random
import re
from collections import defaultdict

SP_DEFAULT = "/work/gfsi/ufsi0002/bs2026-raptor/data/splits"
SPLITS = ("train", "val", "test")
RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 20260905          # ⚠️ 分割は全乱数シードで固定するので、ここは実験の seed とは別物

RULES = ("v1", "v2")

RE_CROP = re.compile(r"^(?:AAS\d*|UDS)_[a-z]+-(.+?)-\d+-\d+-\d+-\d+\.JPG$", re.I)
RE_FRAME = re.compile(r"^frame_\d+_bb\d+_t[\d.]+_conf[\d.]+\.jpg$", re.I)
RE_INAT = re.compile(r"^inat_(\d+)_[0-9a-f]+\.jpg$", re.I)
RE_EBIRD = re.compile(r"^ebird_(\d+)\.jpg$", re.I)
# ⚠️ make_data_manifest.py:RE_VFRAME と 1 文字も違えてはいけない (群が食い違うと比較が壊れる)
RE_VFRAME = re.compile(r"^([0-9a-f]{8})__frame_(\d+)_bb\d+_t([\d.]+)_conf[\d.]+\.jpg$", re.I)


def group_of(rel_path, species_key, rule="v1"):
    """rel_path から group ID を決める。種をまたぐ結合が起きないよう species を必ず前置する.

    rule="v2" のときだけ `<動画ID8桁>__frame_...jpg` を動画 ID ごとに 1 group へまとめる。
    それ以外は v1 と完全に同一 (make_data_manifest.py:group_v2 と同じ構造)。
    """
    name = os.path.basename(rel_path)
    if rule == "v2":
        m = RE_VFRAME.match(name)
        if m:
            return "%s/VIDEO/%s" % (species_key, m.group(1))
    m = RE_CROP.match(name)
    if m:
        return "%s/CROP/%s" % (species_key, m.group(1))
    if RE_FRAME.match(name):
        return "%s/VIDEO" % species_key
    m = RE_INAT.match(name)
    if m:
        return "%s/INAT/%s" % (species_key, m.group(1))
    m = RE_EBIRD.match(name)
    if m:
        return "%s/EBIRD/%s" % (species_key, m.group(1))
    return "%s/SINGLE/%s" % (species_key, rel_path)


def load(sp_dir):
    rows = {}
    for s in SPLITS:
        with open(os.path.join(sp_dir, s + ".csv")) as f:
            rows[s] = list(csv.DictReader(f))
    return rows


def report(rows, rule="v1"):
    """現行分割で、同一 group が複数の split にまたがっている枚数を数える."""
    where = defaultdict(set)
    for s in SPLITS:
        for r in rows[s]:
            where[group_of(r["rel_path"], r["species_key"], rule)].add(s)
    print("=== 現行分割の漏洩 (group が複数 split にまたがる) ===")
    if rule != "v1":
        print("  群規則: %s (動画 ID 接頭辞を 1 group にまとめる)" % rule)
    print("  group 総数: %d" % len(where))
    span = {g: v for g, v in where.items() if len(v) > 1}
    print("  またがる group: %d" % len(span))
    for s in SPLITS:
        n = sum(1 for r in rows[s]
                if group_of(r["rel_path"], r["species_key"], rule) in span)
        print("  %-5s の該当枚数: %4d / %4d  (%.2f%%)" % (s, n, len(rows[s]), 100.0 * n / len(rows[s])))
    big = sorted(span.items(), key=lambda kv: -len(kv[1]))[:5]
    for g, v in big:
        print("    例 %-46s %s" % (g, sorted(v)))
    return span


def straddling(out, rule):
    """分割 out の中で、同一 group が複数 split にまたがっているものを返す."""
    where = defaultdict(set)
    for s in SPLITS:
        for r in out[s]:
            where[group_of(r["rel_path"], r["species_key"], rule)].add(s)
    return [g for g, v in where.items() if len(v) > 1]


def build(rows, out_dir, rule="v1"):
    """group 単位の層化分割を作る."""
    by_species = defaultdict(lambda: defaultdict(list))
    for s in SPLITS:
        for r in rows[s]:
            g = group_of(r["rel_path"], r["species_key"], rule)
            by_species[r["species_key"]][g].append(r)

    rng = random.Random(SEED)
    assign = {}
    print("\n=== group 単位の層化分割 ===")
    print("%-12s %6s %6s | %6s %6s %6s" % ("種", "group", "枚数", "train", "val", "test"))
    for sp in sorted(by_species):
        groups = sorted(by_species[sp].items(), key=lambda kv: (-len(kv[1]), kv[0]))
        total = sum(len(v) for _, v in groups)
        target = {s: RATIO[s] * total for s in SPLITS}
        cur = {s: 0 for s in SPLITS}
        # 大きい group から、目標に対する不足が最大の split へ入れる (貪欲)
        head, tail = groups[:1], groups[1:]
        rng.shuffle(tail)
        for g, items in head + tail:
            s = max(SPLITS, key=lambda s: target[s] - cur[s])
            assign[g] = s
            cur[s] += len(items)
        print("%-12s %6d %6d | %6d %6d %6d" % (sp, len(groups), total,
                                               cur["train"], cur["val"], cur["test"]))

    out = {s: [] for s in SPLITS}
    for sp in by_species:
        for g, items in by_species[sp].items():
            out[assign[g]].extend(items)

    # 検証: group が 1 つの split にしか現れないこと
    # ⚠️ --rule v2 で作った分割は **v2 規則でも v1 規則でも**またがらないことを両方確認する
    #    (v2 は v1 の粗化なので理論上は自明だが、規則を書き換えたときに気づけるようにする)
    bad = straddling(out, rule)
    print("\n=== 検証 ===")
    print("  またがる group: %d 件 %s" % (len(bad), "(OK)" if not bad else "⛔ 不正"))
    bad_other = []
    if rule != "v1":
        for other in [r for r in RULES if r != rule]:
            b = straddling(out, other)
            bad_other += b
            print("  またがる group (%s 規則でも検証): %d 件 %s"
                  % (other, len(b), "(OK)" if not b else "⛔ 不正"))
    n_all = sum(len(out[s]) for s in SPLITS)
    n_in = sum(len(rows[s]) for s in SPLITS)
    print("  枚数: %d (入力 %d) %s" % (n_all, n_in, "(OK)" if n_all == n_in else "⛔ 欠落"))
    for s in SPLITS:
        print("  %-5s %5d 枚 (%.1f%%)" % (s, len(out[s]), 100.0 * len(out[s]) / n_all))
    if bad or bad_other or n_all != n_in:
        raise SystemExit("[abort] 検証に失敗した")

    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        p = os.path.join(out_dir, s + ".csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rel_path", "species_idx", "species_key"])
            w.writeheader()
            for r in sorted(out[s], key=lambda r: r["rel_path"]):
                w.writerow({k: r[k] for k in w.fieldnames})
        print("  書き出し: %s (%d 行)" % (p, len(out[s])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=SP_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rule", default="v1", choices=list(RULES),
                    help="群の定義。v1=A-3 で使った規則 (既定・従来どおり)、"
                         "v2=動画 ID 接頭辞を 1 group にまとめる修正規則")
    a = ap.parse_args()
    rows = load(a.splits)
    report(rows, a.rule)
    if a.out:
        build(rows, a.out, a.rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
