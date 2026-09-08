#!/usr/bin/env bash
# Phase 3: Orin Nano の消費電力・メモリを tegrastats で実測する.
#
# 全 56 構成では時間がかかるので、各モデルの代表 3 解像度 (16 / 112 / 224) = 12 構成に絞る。
# エンジンは既にビルド済みなので --loadEngine で読み込み、推論中の電力を記録する。
#
# 測定条件は運用と同じ 15W モード・DVFS 有効のまま (jetson_clocks は使わない)。
# tegrastats は別プロセスだが CPU 負荷が小さく、計測を汚さないことを事前に確認する。
#
# 起動: bash run_orin_power.sh
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
mkdir -p "$E/results_power" "$E/logs_power"
cd "$E" || exit 1

echo "===== 電力測定開始 $(date '+%H:%M:%S') ====="
echo "power mode: $(nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1)"

# アイドル時のベースラインを先に取る (推論分の増分を出すため)
echo "--- アイドル 20 秒 ---"
timeout 20 tegrastats --interval 500 > "$E/logs_power/idle.log" 2>&1 || true

for mo in mnv4 effb0 resnet50 vit_small; do
    for r in 16 112 224; do
        tag="${mo}_r${r}"
        eng="$E/engines/${tag}.engine"
        [ -f "$eng" ] || { echo "  [miss] $tag"; continue; }
        echo "--- $tag ---"
        tegrastats --interval 500 > "$E/logs_power/${tag}.log" 2>&1 &
        TG=$!
        sleep 2
        "$TRTEXEC" --loadEngine="$eng" --warmUp=3000 --iterations=1500 --avgRuns=100 \
            > "$E/logs_power/trt_${tag}.log" 2>&1
        sleep 1
        kill "$TG" 2>/dev/null
        wait "$TG" 2>/dev/null
    done
done

echo "===== 集計 ====="
python3 - << "PYEOF"
import glob, json, os, re
E = "/home1/gfsi/ufsi0002/bs2026-resolution-edge"

def parse(path):
    """tegrastats の行から電力 (mW) と RAM (MB) を取る。
    Orin の書式例: RAM 1234/6480MB ... VDD_IN 4321mW/4000mW VDD_CPU_GPU_CV 1234mW/1200mW VDD_SOC 800mW/790mW
    """
    rails = {}
    ram = []
    for line in open(path, errors="ignore"):
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            ram.append(int(m.group(1)))
        for name, cur in re.findall(r"(VDD_\w+|POM_\w+) (\d+)mW", line):
            rails.setdefault(name, []).append(int(cur))
    out = {"n_samples": len(ram), "ram_used_MB_mean": round(sum(ram) / len(ram), 1) if ram else None}
    for k, v in rails.items():
        out[k + "_mW_mean"] = round(sum(v) / len(v), 1)
        out[k + "_mW_max"] = max(v)
    return out

idle = parse(os.path.join(E, "logs_power/idle.log"))
res = {"idle": idle, "configs": {}}
for f in sorted(glob.glob(os.path.join(E, "logs_power/*.log"))):
    b = os.path.basename(f)
    if b.startswith("trt_") or b == "idle.log":
        continue
    tag = b[:-4]
    rec = parse(f)
    # アイドルからの増分 (推論に起因する電力)
    for k in list(rec):
        if k.endswith("_mW_mean") and k in idle and idle[k] is not None:
            rec[k.replace("_mW_mean", "_mW_delta")] = round(rec[k] - idle[k], 1)
    res["configs"][tag] = rec
json.dump(res, open(os.path.join(E, "results_power/power.json"), "w"), indent=2, ensure_ascii=False)

print("アイドル: " + ", ".join("%s=%.0f mW" % (k.replace("_mW_mean", ""), v)
                              for k, v in idle.items() if k.endswith("_mW_mean")))
print("%-18s %10s %10s %10s" % ("config", "VDD_IN(mW)", "増分(mW)", "RAM(MB)"))
for tag, r in sorted(res["configs"].items()):
    print("%-18s %10.0f %10.0f %10.0f" % (
        tag, r.get("VDD_IN_mW_mean", -1), r.get("VDD_IN_mW_delta", -1), r.get("ram_used_MB_mean", -1)))
print("[saved] results_power/power.json")
PYEOF
touch "$E/logs_power/power.done"
