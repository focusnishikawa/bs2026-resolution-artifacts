#!/usr/bin/env bash
# ============================================================================
# 条件 A の DINOv2-L 評価を、30 シードすべてそろうまで繰り返す
# ============================================================================
#
# ⚠️ 背景 (2026-09-07 実測):
#    fill_group_condA_dinov2.sh は 30 シードを **1 巡するだけ**で、足りなければ
#    rc=1 で終わる。ところが eval_sweep.py が **約 50% の確率で rc=139
#    (Segmentation fault)** で落ちることが分かった。1 巡目の実測は
#    22 シード試行して 11 シード失敗である。
#
# ⚠️ 落ち方の特徴 (失敗ログで確認済み):
#    - 成功は 42-43 秒で 3 構成すべて評価 → 16/16
#    - 失敗は 22-23 秒で **1 件も書かずに落ちる** → 13/16 のまま
#      crop 1,876 枚の読み込み直後、最初の `[A] dinov2_l res=64` を出す前に落ちる
#    - 特定シードに固有ではない (同じ seed が巡によって成功も失敗もする)
#    → S15 が本線 (fill_group_eval.sh) で見つけた UCX の Caught signal 11 と
#      同じ現象。**逐次実行でも起きる**ので、リトライで押し切るしかない。
#
# ⭐ 再実行が安全な理由:
#    1. eval_sweep.py に --skip_existing があるので成功済みの構成は再計算しない
#    2. 失敗は 0/3 書き込みなので、途中まで書けた壊れた JSON が残らない
#
# 使い方:
#    cd /work/gfsi/ufsi0002/bs2026-resolution
#    setsid nohup bash train/retry_group_condA_dinov2.sh \
#        > logs/fill_condA_retry.log 2>&1 < /dev/null &
# ============================================================================
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SPG=/work/gfsi/ufsi0002/bs2026-raptor/data/splits_group
V2L="64 144 160"
SEEDS=$(seq 42 71)
NEED_A=16
MAX_ROUND=12

cd "$PROJ" || exit 1
[ -d "$SPG" ] || { echo "[abort] group split が無い: $SPG"; exit 1; }

APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1"
APP="$APP --env UCX_HANDLE_ERRORS=none --bind /work --bind /home1 $SIF"

# ⚠️ 判定は必ず厳密な正規表現で行う。`ls | grep -c '\.json$'` だと
#    summary*.json まで数えてしまい 17/16 のような値になる (S15 の教訓)。
count_json() {
    ls "$1" 2>/dev/null | grep -cE '^(resnet50|dinov2_l|vit_small)_r[0-9]+\.json$'
}

running_1st() {
    ps -u "$(id -un)" -o cmd= 2>/dev/null | grep -c '[f]ill_group_condA_dinov2\.sh'
}

T0=$(date +%s)
echo "===== 条件 A DINOv2-L のリトライ 開始 $(date '+%F %T') ====="

# ---- 1. 1 巡目 (fill_group_condA_dinov2.sh) の終了を待つ (最大 60 分) ----
#    GPU を取り合わないため。実行中のスクリプトには一切触らない。
w=0
while [ "$(running_1st)" -gt 0 ]; do
    [ "$w" -ge 60 ] && { echo "[abort] 60 分待っても 1 巡目が終わらない"; exit 1; }
    [ "$w" -eq 0 ] && echo "[wait] 1 巡目 (fill_group_condA_dinov2.sh) の終了を待つ"
    sleep 60; w=$((w+1))
done
echo "[$(date '+%F %T')] 先行ジョブなし。リトライに入る"

# ---- 2. そろうまで巡回する ----
for r in $(seq 1 "$MAX_ROUND"); do
    todo=""
    for s in $SEEDS; do
        n=$(count_json "$PROJ/results/G_condA_s${s}")
        [ "$n" -ge "$NEED_A" ] || todo="$todo $s"
    done
    if [ -z "$todo" ]; then
        echo "[$(date '+%F %T')] 全 30 シードが ${NEED_A}/16 でそろった"
        break
    fi
    n_todo=$(echo $todo | wc -w)
    echo "[$(date '+%F %T')] --- 巡回 ${r} / 未了 ${n_todo} 件:${todo} ---"

    n_ok=0
    for s in $todo; do
        out="$PROJ/results/G_condA_s${s}"
        [ -s "$PROJ/models_group_s${s}/dinov2_l_r224/best.pt" ] || {
            echo "    [skip] s${s} (dinov2_l_r224 の重みが無い)"; continue; }
        mkdir -p "$out"
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=0 $APP python3 train/eval_sweep.py \
            --mode A --models dinov2_l --resolutions $V2L \
            --models_dir "$PROJ/models_group_s${s}" --output_dir "$out" \
            --data_root "$DR" --splits_dir "$SPG" \
            --skip_existing --summary_name summary_fill_dinov2_condA.json \
            >> "logs/fill_ev_A_${s}_dinov2_l.log" 2>&1
        # ⚠️ rc を printf の引数に直接書かないこと。引数中の $(date ...) が先に
        #    評価されて $? を上書きし、常に 0 になる (S15 が踏んだ)。
        rc=$?
        el=$(( $(date +%s)-t0 ))
        nj=$(count_json "$out")
        [ "$rc" -eq 0 ] && n_ok=$((n_ok+1))
        printf "    [%s] condA s%-2s rc=%-3d %3ds -> %s/16\n" \
               "$(date '+%H:%M:%S')" "$s" "$rc" "$el" "$nj"
    done
    echo "[$(date '+%F %T')] 巡回 ${r} 終了: 今回 rc=0 が ${n_ok} / ${n_todo} 件"
    sleep 30
done

# ---- 3. 完了判定は**データを数えて**行う (マーカーや rc を信じない) ----
ok=0; short=""
for s in $SEEDS; do
    n=$(count_json "$PROJ/results/G_condA_s${s}")
    if [ "$n" -ge "$NEED_A" ]; then ok=$((ok+1)); else short="$short s${s}(${n})"; fi
done
echo "===== 終了 $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/60 )) 分 ====="
echo "  条件 A が ${NEED_A}/16 の seed: ${ok}/30"
[ -n "$short" ] && echo "  未了:${short}"

if [ "$ok" -ge 30 ]; then
    touch "$PROJ/logs/fill_condA_dinov2.done"
    echo "  マーカー logs/fill_condA_dinov2.done を立てた"
    exit 0
fi
echo "⛔ そろわなかった。logs/fill_ev_A_*_dinov2_l.log を見ること"
exit 1
