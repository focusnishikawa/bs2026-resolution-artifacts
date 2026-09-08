#!/usr/bin/env bash
# NVDEC の持続スループットと、1 ユニット分 (4 ストリーム) の同時デコードを測る。
#
# nvdec_bench.py が測るのは「1 フレームを BGR ホストメモリへ取り出す時間」である。
# 本スクリプトはそれと別に、
#   (c)  NVDEC のみ (NVMM 内で完結)      = 検出器を NVMM 直結にした場合の下限
#   (c2) NVDEC -> BGRx (色変換のみ)       = videoconvert を省いた場合
#   (c3) NVDEC -> BGR  (既存コード互換)
#   (e)  4 ストリーム同時                 = 1 ユニット (4 台 x 10 fps = 40 fps) を捌けるか
# を測る。
#
# ⚠️ fpsdisplaysink の average-fps は `gst-launch -q` だと出ないので使わない。
#    長短 2 本の壁時計時間から回帰してパイプライン起動時間を差し引く:
#        t(N) = a + N/fps   →   fps = (N_long - N_short) / (t_long - t_short)
#
# ⚠️ bash で起動すること (zsh 不可)。
# usage: bash nvdec_streams.sh <長い動画> <短い動画> <長さのフレーム数> <短さのフレーム数> [出力JSON]
set -u

LONG="${1:?長い動画}"
SHORT="${2:?短い動画}"
NLONG="${3:?長い方のフレーム数}"
NSHORT="${4:?短い方のフレーム数}"
OUT="${5:-nvdec_streams.json}"
for f in "$LONG" "$SHORT"; do [ -f "$f" ] || { echo "[NG] 動画が無い: $f"; exit 1; }; done

SINK_ONLY='fakesink sync=false'
SINK_BGRX='nvvidconv ! video/x-raw,format=BGRx ! fakesink sync=false'
SINK_BGR='nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! fakesink sync=false'

# 1 本走らせて壁時計秒を返す
timeit() {
    local video="$1" sink="$2"
    local t0 t1
    t0=$(python3 -c 'import time;print(time.perf_counter())')
    gst-launch-1.0 -q filesrc location="$video" ! qtdemux ! h264parse ! nvv4l2decoder ! ${sink} > /dev/null 2>&1
    t1=$(python3 -c 'import time;print(time.perf_counter())')
    awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b-a}'
}

# 長短 2 本から起動時間を除いた fps を出す (各 3 回の中央値を使う)
measure() {
    local tag="$1" sink="$2"
    local L=() S=()
    for i in 1 2 3; do L+=("$(timeit "$LONG" "$sink")"); done
    for i in 1 2 3; do S+=("$(timeit "$SHORT" "$sink")"); done
    python3 - "$tag" "$NLONG" "$NSHORT" "${L[@]}" "${S[@]}" <<'PYEOF'
import sys, statistics as st, json
tag, nl, ns = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
v = [float(x) for x in sys.argv[4:]]
L, S = v[:3], v[3:]
tl, ts = st.median(L), st.median(S)
fps = (nl - ns) / (tl - ts) if tl > ts else float('nan')
startup = ts - ns / fps if fps == fps else float('nan')
print(json.dumps({"tag": tag, "t_long": round(tl,4), "t_short": round(ts,4),
                  "fps": round(fps,2), "ms_per_frame": round(1000/fps,3),
                  "startup_s": round(startup,4),
                  "runs_long": L, "runs_short": S}, ensure_ascii=False))
PYEOF
}

TMP=$(mktemp -d)
echo "=== (c) NVDEC のみ (NVMM 内で完結) ==="
measure nvdec_only "$SINK_ONLY" | tee "$TMP/c.json"
echo ""
echo "=== (c2) NVDEC -> BGRx (nvvidconv のみ) ==="
measure nvdec_bgrx "$SINK_BGRX" | tee "$TMP/c2.json"
echo ""
echo "=== (c3) NVDEC -> BGR (videoconvert 込み・既存コード互換) ==="
measure nvdec_bgr "$SINK_BGR" | tee "$TMP/c3.json"

echo ""
echo "=== (e) 4 ストリーム同時 (NVDEC のみ・1 ユニット分) ==="
T0=$(python3 -c 'import time;print(time.perf_counter())')
for s in 1 2 3 4; do
    ( gst-launch-1.0 -q filesrc location="$LONG" ! qtdemux ! h264parse ! nvv4l2decoder ! ${SINK_ONLY} > /dev/null 2>&1 ) &
done
wait
T1=$(python3 -c 'import time;print(time.perf_counter())')
ELAPSED=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.4f", b-a}')
echo "  4 本の壁時計: ${ELAPSED} s"

python3 - "$OUT" "$TMP" "$ELAPSED" "$NLONG" <<'PYEOF'
import json, sys, os
out, tmp, elapsed, nlong = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
d = {"note": "NVDEC の持続スループット。起動時間は長短2本の回帰で除去した。"
             "(c) は NVMM 内で完結、(c2) は BGRx まで、(c3) は BGR まで。"
             "(e) は 4 ストリーム同時で 1 ユニットの最低目標は 40 fps"}
for key, f in (("nvdec_only","c.json"),("nvdec_bgrx","c2.json"),("nvdec_bgr","c3.json")):
    p = os.path.join(tmp, f)
    if os.path.exists(p):
        d[key] = json.load(open(p))
total = 4*nlong/elapsed
d["four_streams"] = {"elapsed_s": elapsed, "total_fps": round(total,2),
                     "per_stream_fps": round(total/4,2), "target_fps": 40.0,
                     "meets_target": total >= 40.0,
                     "note": "起動時間を含む保守側の値"}
json.dump(d, open(out,"w"), indent=2, ensure_ascii=False)
print("  合計 %.1f fps (1 本あたり %.1f fps) / 目標 40 fps -> %s"
      % (total, total/4, "達成" if total>=40 else "未達"))
print("\n[saved] %s" % out)
PYEOF
rm -rf "$TMP"
