#!/usr/bin/env bash
# ViT-S/16 の Orin latency が解像度に対して単調でない原因を切り分ける.
#
# 症状: N=64 だけ 3.927 ms と、前後 (N=48 2.687 / N=80 2.826) から大きく外れる。
#       他の 3 モデルは単調違反ゼロで、ViT-S だけ 4 件の違反がある。
#
# 見えている手掛かり:
#   - ONNX は全水準 86.5 MB で同一なのに、**engine は N=64 だけ 86.3 MB** (他は 44 MB)。
#     44 MB は FP16 の重み、86 MB は FP32 のまま = ビルド時に FP16 経路が選ばれなかった疑い
#   - 1 回の計測内のばらつき (max-min)/min が ViT-S は 25-34% と大きい (mnv4 は 3-14%)。
#     TRT はビルド時にカーネルを実測で選ぶので、計測が揺れるとタクティク選択も揺れる
#
# そこで 2 つを分けて確かめる:
#   (1) 既存 engine を --loadEngine で 3 回ずつ測り直す -> 計測ノイズの大きさと再現性
#   (2) N=64 を再ビルドする -> engine サイズが 44 MB に戻り latency も揃うか
#
# 起動 (Orin):
#   setsid nohup bash remeasure_vit.sh > logs/remeasure_vit_master.log 2>&1 &
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
TRTEXEC=/usr/src/tensorrt/bin/trtexec
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
OUT="$E/results/remeasure_vit.csv"
mkdir -p "$E/logs_remeasure"
rm -f "$E/logs/remeasure_vit.done"

n=$(pgrep -c trtexec 2>/dev/null || true)
if [ "${n:-0}" -gt 0 ]; then echo "[abort] trtexec が $n 本動作中"; exit 1; fi

mean_of() {  # trtexec ログから GPU Compute Time の mean / median を取る
    grep -aoE "GPU Compute Time: min = [0-9.]+ ms, max = [0-9.]+ ms, mean = [0-9.]+ ms, median = [0-9.]+" "$1" \
        | tail -1 | grep -oE "[0-9.]+" | tr '\n' ' '
}

echo "res,rep,min,max,mean,median" > "$OUT"
echo "===== (1) 既存 engine を 3 回ずつ再計測 ====="
for r in $RES; do
    eng="$E/engines/vit_small_r${r}.engine"
    [ -f "$eng" ] || { echo "  [miss] r$r"; continue; }
    sz=$(stat -c %s "$eng" | awk '{printf "%.1f", $1/1e6}')
    line="  r%-4s engine=${sz}MB "
    printf "  r%-4s engine=%6s MB " "$r" "$sz"
    for rep in 1 2 3; do
        log="$E/logs_remeasure/vit_small_r${r}_rep${rep}.log"
        "$TRTEXEC" --loadEngine="$eng" --warmUp=2000 --iterations=300 --avgRuns=100 \
            > "$log" 2>&1
        v=$(mean_of "$log")
        set -- $v
        echo "$r,$rep,$1,$2,$3,$4" >> "$OUT"
        printf "rep%d=%s " "$rep" "$3"
    done
    echo
done

echo "===== (2) N=64 を再ビルド ====="
eng2="$E/engines/vit_small_r64_rebuild.engine"
rm -f "$eng2"
"$TRTEXEC" --onnx="$E/onnx/vit_small_r64.onnx" --saveEngine="$eng2" --fp16 \
    --workspace=1024 --warmUp=2000 --iterations=300 --avgRuns=100 \
    > "$E/logs_remeasure/rebuild_vit_small_r64.log" 2>&1
rc=$?
sz=$(stat -c %s "$eng2" 2>/dev/null | awk '{printf "%.1f", $1/1e6}')
echo "  rc=$rc engine=${sz} MB (元の engine は 86.3 MB / 他水準は 44 MB)"
v=$(mean_of "$E/logs_remeasure/rebuild_vit_small_r64.log")
echo "  再ビルド後 min/max/mean/median = $v"
for rep in 1 2 3; do
    log="$E/logs_remeasure/vit_small_r64_rebuild_rep${rep}.log"
    "$TRTEXEC" --loadEngine="$eng2" --warmUp=2000 --iterations=300 --avgRuns=100 > "$log" 2>&1
    v=$(mean_of "$log")
    set -- $v
    echo "64rebuild,$rep,$1,$2,$3,$4" >> "$OUT"
    echo "  rep$rep mean=$3"
done

echo "===== 完了 ====="
touch "$E/logs/remeasure_vit.done"
