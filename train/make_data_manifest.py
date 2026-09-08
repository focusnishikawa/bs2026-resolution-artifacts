#!/usr/bin/env python3
"""データの出所 manifest を作る (再レビュー B-2「再現性のための表と公開物」).

画像そのものは再配布しない (iNaturalist は写真ごとにライセンスが異なり、Macaulay Library は
再配布不可、自前撮影は著者所有)。代わりに **取得元 ID・URL・sha256・群 ID・両分割の所属**を
1 枚 1 行で出し、第三者が同じ画像集合を組み直せるようにする。

⚠️ 群 ID は 2 種類出す。
   `group_id`    = 実験で実際に使った規則 (train/make_group_split.py:group_of)
   `group_id_v2` = 動画フレームの接頭辞 8 桁 (動画 ID) を使う修正規則
   `bird_frames_pure_oowashi_20260504/` の 1,000 枚は
   `<動画ID8桁>__frame_...jpg` という名前なので **v1 の RE_FRAME に一致せず 1 枚 1 群**に
   なっていた。v2 で数え直すと同一動画が分割をまたぐ枚数が出る (論文の限界節に書く値)。

usage:
  python3 train/make_data_manifest.py \
      --index      ../bs2026-raptor/data/raptor_subset_index.csv \
      --splits     ../bs2026-raptor/data/splits \
      --splits-group data/splits_group \
      --sha256     <sha256sum の出力> \
      --out-csv    data/manifest.csv \
      --out-json   results/data_provenance.json
"""
import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test")

# --- 出所の判定 (ファイル名優先、無ければ source_tag) ---------------------------
RE_INAT = re.compile(r"^inat_(\d+)_[0-9a-f]+\.jpg$", re.I)
RE_EBIRD = re.compile(r"^ebird_(\d+)\.jpg$", re.I)
RE_WM = re.compile(r"^wm_([0-9a-f]{12})\.(?:jpg|png)$", re.I)
RE_CROP = re.compile(r"^(?:AAS\d*|UDS)_[a-z]+-(.+?)-\d+-\d+-\d+-\d+\.JPG$", re.I)
RE_FRAME = re.compile(r"^frame_\d+_bb\d+_t[\d.]+_conf[\d.]+\.jpg$", re.I)
RE_VFRAME = re.compile(r"^([0-9a-f]{8})__frame_(\d+)_bb\d+_t([\d.]+)_conf[\d.]+\.jpg$", re.I)

URL_INAT = "https://www.inaturalist.org/observations/%s"
URL_EBIRD = "https://macaulaylibrary.org/asset/%s"
URL_BIRDS525 = "https://www.kaggle.com/datasets/gpiosenka/100-bird-species"
# Wikimedia はファイル名が内容の sha1 先頭 12 桁なので、取得手順 (クロールログの URL 一覧を
# 順に取得して sha1 を取る) を公開すれば 1 対 1 に対応づけられる。個々の URL は
# .wm_crawl_log.json (種ごとの取得順) に入っている。
URL_WM_LOG = "wikimedia_crawl_log:%s"


def classify(rel_path, source_tag):
    """(source_kind, source_id, source_url) を返す."""
    name = os.path.basename(rel_path)
    m = RE_INAT.match(name)
    if m:
        return "inaturalist", m.group(1), URL_INAT % m.group(1)
    m = RE_EBIRD.match(name)
    if m:
        return "macaulay", m.group(1), URL_EBIRD % m.group(1)
    m = RE_WM.match(name)
    if m:
        return "wikimedia", m.group(1), URL_WM_LOG % m.group(1)
    m = RE_VFRAME.match(name)
    if m:
        return "own_video", m.group(1), ""
    if RE_FRAME.match(name):
        return "own_video_noid", "", ""
    m = RE_CROP.match(name)
    if m:
        return "own_photo", m.group(1), ""
    if source_tag == "BIRDS525":
        return "birds525", os.path.splitext(name)[0], URL_BIRDS525
    return "legacy_internal", "", ""


def group_v1(rel_path, species_key):
    """実験で使った群規則 (make_group_split.py:group_of と同一)."""
    name = os.path.basename(rel_path)
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


def group_v2(rel_path, species_key):
    """動画 ID を使う修正規則. 接頭辞つきフレームだけ v1 と異なる."""
    m = RE_VFRAME.match(os.path.basename(rel_path))
    if m:
        return "%s/VIDEO/%s" % (species_key, m.group(1))
    return group_v1(rel_path, species_key)


def load_split(d):
    """rel_path -> split の対応と、species_idx・species_key を返す.

    ⚠️ species_key は分割 CSV の値を正とする。`rel_path` の第 2 要素から導くと
    `inaturalist/additional_ebird/Ootaka_.../ebird_*.jpg` の 30 枚 (画像ルートが違う) で
    "additional" になり、群が種をまたいで分かれてしまう。
    """
    where, idx, key = {}, {}, {}
    for s in SPLITS:
        path = os.path.join(d, s + ".csv")
        with open(path) as f:
            for r in csv.DictReader(f):
                where[r["rel_path"]] = s
                idx[r["rel_path"]] = r["species_idx"]
                key[r["rel_path"]] = r["species_key"]
    return where, idx, key


def straddle_stats(rows, keyfn, split_field):
    """群がいくつ分割をまたぐかと、該当枚数を 2 通りの定義で数える.

    ⚠️ 数え方が 2 つあり、混ぜてはいけない。
      `in_straddling_group` = またがる群に属する枚数 (**論文が報告している定義**。
                              make_group_split.py:report と同じ)
      `sharing_train`       = そのうち train も含む群に属する枚数 (より厳しい定義)
    """
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r[split_field]:
            per[keyfn(r)][r[split_field]] += 1
    span = {g: c for g, c in per.items() if len(c) > 1}
    out = {"groups": len(per), "straddling_groups": len(span)}
    for s in SPLITS:
        out["in_straddling_group_%s" % s] = sum(c[s] for c in span.values())
        out["sharing_train_%s" % s] = sum(c[s] for c in span.values() if "train" in c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="../bs2026-raptor/data/raptor_subset_index.csv")
    ap.add_argument("--splits", default="../bs2026-raptor/data/splits")
    ap.add_argument("--splits-group", default="data/splits_group")
    ap.add_argument("--sha256", default="", help="sha256sum の出力 (省略時は空欄)")
    ap.add_argument("--out-csv", default="data/manifest.csv")
    ap.add_argument("--out-json", default="results/data_provenance.json")
    a = ap.parse_args()

    tag = {}
    with open(a.index) as f:
        for r in csv.DictReader(f):
            tag[r["rel_path"]] = r["source_tag"]

    img_where, img_idx, img_key = load_split(a.splits)
    grp_where, _, grp_key = load_split(a.splits_group)

    sha = {}
    if a.sha256:
        with open(a.sha256) as f:
            for line in f:
                h, _, p = line.strip().partition("  ")
                if p:
                    sha[p] = h

    rows = []
    for rel in sorted(set(img_where) | set(grp_where) | set(tag)):
        species_key = img_key.get(rel) or grp_key.get(rel) or rel.split("/")[1].split("_")[0]
        st = tag.get(rel, "")
        kind, sid, url = classify(rel, st)
        rows.append({
            "rel_path": rel,
            "species_key": species_key,
            "species_idx": img_idx.get(rel, ""),
            "source_tag": st,
            "source_kind": kind,
            "source_id": sid,
            "source_url": url,
            "split_image": img_where.get(rel, ""),
            "split_group": grp_where.get(rel, ""),
            "group_id": group_v1(rel, species_key),
            "group_id_v2": group_v2(rel, species_key),
            "sha256": sha.get(rel, ""),
        })

    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- 集計 -----------------------------------------------------------------
    by_kind = Counter(r["source_kind"] for r in rows)
    by_tag = Counter(r["source_tag"] for r in rows)
    with_url = sum(1 for r in rows if r["source_url"])
    with_sha = sum(1 for r in rows if r["sha256"])
    per_species_tag = defaultdict(Counter)
    for r in rows:
        per_species_tag[r["species_key"]][r["source_tag"]] += 1
    per_split_tag = defaultdict(Counter)
    for r in rows:
        if r["split_image"]:
            per_split_tag[r["split_image"]][r["source_tag"]] += 1

    # 動画の時間間隔 (同一動画内で隣り合うフレームの秒差)
    vid = defaultdict(list)
    for r in rows:
        m = RE_VFRAME.match(os.path.basename(r["rel_path"]))
        if m:
            vid[m.group(1)].append(float(m.group(3)))
    gaps = []
    for v in vid.values():
        v.sort()
        gaps += [round(v[i + 1] - v[i], 2) for i in range(len(v) - 1)]

    out = {
        "total_images": len(rows),
        "counts_by_source_kind": dict(by_kind),
        "counts_by_source_tag": dict(by_tag),
        "counts_by_species_and_tag": {k: dict(v) for k, v in per_species_tag.items()},
        "counts_by_split_and_tag": {k: dict(v) for k, v in per_split_tag.items()},
        "identifiability": {
            "with_source_url": with_url,
            "without_source_url": len(rows) - with_url,
            "with_sha256": with_sha,
            "without_sha256": len(rows) - with_sha,
        },
        "split_sizes": {
            "image_wise": dict(Counter(r["split_image"] for r in rows if r["split_image"])),
            "group": dict(Counter(r["split_group"] for r in rows if r["split_group"])),
        },
        "group_rule_v1_used_in_experiments": {
            "image_wise_split": straddle_stats(rows, lambda r: r["group_id"], "split_image"),
            "group_split": straddle_stats(rows, lambda r: r["group_id"], "split_group"),
        },
        "group_rule_v2_video_id_aware": {
            "image_wise_split": straddle_stats(rows, lambda r: r["group_id_v2"], "split_image"),
            "group_split": straddle_stats(rows, lambda r: r["group_id_v2"], "split_group"),
        },
        "video_frames": {
            "videos": len(vid),
            "frames": sum(len(v) for v in vid.values()),
            "frames_per_video_median": statistics.median([len(v) for v in vid.values()]) if vid else 0,
            "adjacent_gap_seconds": {
                "min": min(gaps) if gaps else 0,
                "median": statistics.median(gaps) if gaps else 0,
                "mean": round(statistics.mean(gaps), 2) if gaps else 0,
                "max": max(gaps) if gaps else 0,
                "pairs_within_1s": sum(1 for g in gaps if g <= 1.0),
                "pairs_within_5s": sum(1 for g in gaps if g <= 5.0),
                "pairs": len(gaps),
            },
        },
    }
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("manifest: %s (%d 行)" % (a.out_csv, len(rows)))
    print("出所: " + " / ".join("%s=%d" % kv for kv in by_kind.most_common()))
    print("URL あり %d / なし %d ・sha256 あり %d / なし %d"
          % (with_url, len(rows) - with_url, with_sha, len(rows) - with_sha))
    for name in ("group_rule_v1_used_in_experiments", "group_rule_v2_video_id_aware"):
        s = out[name]["group_split"]
        print("%-34s group split: 群 %d ・またがる群 %d ・該当 test %d / val %d ・うち train 共有 test %d / val %d"
              % (name, s["groups"], s["straddling_groups"],
                 s["in_straddling_group_test"], s["in_straddling_group_val"],
                 s["sharing_train_test"], s["sharing_train_val"]))


if __name__ == "__main__":
    main()
