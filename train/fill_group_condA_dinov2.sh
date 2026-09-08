#!/usr/bin/env bash
# 査読指摘 A-3 の続き: 条件 A の DINOv2-L を可能にするため dinov2_l_r224 を
# group split で 30 シード追加学習し、条件 A を 16/16 で完備させる (hpciaiss1 側).
#
# ⛔ なぜ要るのか: 境界 16 構成に DINOv2-L の N=224 を入れていなかったため、
#    条件 A (224 学習済みの重みを低解像度で評価する劣化耐性) の元モデルが存在せず、
#    eval_sweep.py が `[miss] dinov2_l_r64: .../dinov2_l_r224/best.pt が無い` で
#    3 構成ともスキップしていた (2026-09-06 に seed 42 の実ログで確認)。
#
# ⚠️ 本線 (run_group_split_boundary.sh) と評価穴埋め (fill_group_eval.sh) の
#    **両方の完了を待ってから**動く。GPU を取り合わないためである。
# ⚠️ 学習の 4 並列は安全 (本線も 4 並列で学習しており、落ちているのは評価だけ)。
#    評価だけは **逐次**にする — 3 プロセス同時起動が UCX の Caught signal 11 を招く。
# ⚠️ dinov2_l_r224 は境界で最も重い構成である。実測 (seed 45) は r64 344s / r144 791s /
#    r160 473s なので、r224 は 900-1200 秒とみておく。30 シード / 4 並列で約 2-3 時間。
#
# 起動:
#   cd /work/gfsi/ufsi0002/bs2026-resolution
#   nohup setsid bash train/fill_group_condA_dinov2.sh > logs/fill_condA_dinov2.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SPG=/work/gfsi/ufsi0002/bs2026-raptor/data/splits_group
V2L="64 144 160"
SEEDS=$(seq 42 71)
NEED_A=16

MASTER_DONE="$PROJ/logs/group_boundary.done"
FILL_DONE="$PROJ/logs/fill_group_eval.done"

cd "$PROJ" || exit 1
[ -d "$SPG" ] || { echo "[abort] group split が無い: $SPG"; exit 1; }

APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1"
APP="$APP --env UCX_HANDLE_ERRORS=none --bind /work --bind /home1 $SIF"

count_json() {
    ls "$1" 2>/dev/null | grep -cE '^(resnet50|dinov2_l|vit_small)_r[0-9]+\.json$'
}

T0=$(date +%s)
echo "===== 条件 A の DINOv2-L 完備 開始 $(date '+%F %T') ====="

# ---- 1. 先行ジョブの完了を待つ (最大 60 時間) ----
w=0
while [ ! -f "$MASTER_DONE" ] || [ ! -f "$FILL_DONE" ]; do
    [ "$w" -ge 3600 ] && { echo "[abort] 60 時間待っても先行ジョブが終わらない"; exit 1; }
    [ $((w % 60)) -eq 0 ] && printf "[wait] %s 本線=%s 穴埋め=%s (%d 分)\n" \
        "$(date '+%F %T')" \
        "$([ -f "$MASTER_DONE" ] && echo 済 || echo 未)" \
        "$([ -f "$FILL_DONE" ] && echo 済 || echo 未)" "$w"
    sleep 60; w=$((w+1))
done
echo "[$(date '+%F %T')] 先行ジョブがそろった。dinov2_l_r224 の学習に入る"

# ---- 2. dinov2_l_r224 を 4 seed 並列で学習 ----
# run_phase2.py は 1 seed につき 1 構成しか無いので GPU を 1 枚しか使わない。
# そこで **seed の方を 4 本並べて** 4 GPU を埋める。
batch=""; i=0; nb=0
flush_batch() {
    [ -z "$batch" ] && return 0
    echo "[$(date '+%H:%M:%S')]   学習バッチ: ${batch}"
    local g=0 s
    for s in $batch; do
        SPLITS_DIR="$SPG" nohup python3 train/run_phase2.py \
            --seed "$s" --models dinov2_l --resolutions 224 \
            --models_dir "models_group_s${s}" --gpus "$g" --num_workers 10 \
            --progress_name "gc_dinov2_224_s${s}.json" \
            >> "logs/fill_condA_train_s${s}.log" 2>&1 &
        g=$((g+1))
    done
    wait
    for s in $batch; do
        if [ -s "$PROJ/models_group_s${s}/dinov2_l_r224/best.pt" ]; then
            echo "    [OK]   s${s} dinov2_l_r224"
        else
            echo "    [FAIL] s${s} dinov2_l_r224 (logs/fill_condA_train_s${s}.log を見ること)"
        fi
    done
    batch=""; i=0
}

for s in $SEEDS; do
    # 既にあるものは学習しない (冪等)
    [ -s "$PROJ/models_group_s${s}/dinov2_l_r224/best.pt" ] && { nb=$((nb+1)); continue; }
    batch="$batch $s"; i=$((i+1))
    [ "$i" -ge 4 ] && flush_batch
done
flush_batch
echo "[$(date '+%F %T')] 学習ずみ ${nb} 件はスキップした"

n_ckpt=0
for s in $SEEDS; do
    [ -s "$PROJ/models_group_s${s}/dinov2_l_r224/best.pt" ] && n_ckpt=$((n_ckpt+1))
done
echo "[$(date '+%F %T')] dinov2_l_r224 の重み: ${n_ckpt}/30"

# ---- 3. 条件 A の DINOv2-L を逐次で評価 ----
echo "[$(date '+%F %T')] 条件 A の DINOv2-L を評価する (逐次)"
for s in $SEEDS; do
    out="$PROJ/results/G_condA_s${s}"
    [ -s "$PROJ/models_group_s${s}/dinov2_l_r224/best.pt" ] || {
        echo "    [skip] s${s} (重みが無い)"; continue; }
    mkdir -p "$out"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 $APP python3 train/eval_sweep.py \
        --mode A --models dinov2_l --resolutions $V2L \
        --models_dir "$PROJ/models_group_s${s}" --output_dir "$out" \
        --data_root "$DR" --splits_dir "$SPG" \
        --skip_existing --summary_name summary_fill_dinov2_condA.json \
        >> "logs/fill_ev_A_${s}_dinov2_l.log" 2>&1
    # ⚠️ rc は printf の引数に直接書かない。引数の中の $(date ...) が先に評価されて
    #    $? を上書きしてしまう (常に 0 になる)。必ず変数へ取ってから使う。
    rc=$?
    printf "    [%s] condA s%-2s dinov2_l rc=%d %3ds -> %s/16\n" \
           "$(date '+%H:%M:%S')" "$s" "$rc" "$(( $(date +%s)-t0 ))" "$(count_json "$out")"
done

# ---- 4. 完了判定は**データを数えて**行う ----
ok=0; short=""
for s in $SEEDS; do
    n=$(count_json "$PROJ/results/G_condA_s${s}")
    if [ "$n" -ge "$NEED_A" ]; then ok=$((ok+1)); else short="$short s${s}(${n})"; fi
done
echo "===== 終了 $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/60 )) 分 ====="
echo "  条件 A が 16/16 の seed: ${ok}/30"
[ -n "$short" ] && echo "  未了:${short}"

if [ "$ok" -ge 30 ]; then
    touch "$PROJ/logs/fill_condA_dinov2.done"
    exit 0
fi
echo "⛔ そろわなかった。logs/fill_condA_train_s*.log と logs/fill_ev_A_*_dinov2_l.log を見ること"
exit 1
