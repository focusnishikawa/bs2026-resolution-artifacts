#!/usr/bin/env bash
# 査読指摘 A-3 (v2 規則): group split v2 の**評価だけ**をやり直して穴を埋める (hpciaiss1 側).
#
# ⛔ 本線 train/run_group_split_boundary_v2.sh は条件ごとに **3 プロセスを同時起動**しており、
#    UCX が `Caught signal 11 (Segmentation fault)` を出して評価プロセスが散発的に全滅する。
#    v1 での実測 (2026-09-06 02:5x・models_group_s*):
#      seed 42 条件A 11/16 ／ seed 44 条件B 14/16 ／ seed 45 条件B 11/16 (条件A 14/16)
#    ⭐ **学習は 16/16 すべて成功していた** (「Phase 2 完了 seed=45: 3 本 / 失敗 0 本」)
#    ので、やり直すのは**評価だけ**でよい。v2 でも同じ現象を前提に置く。
#
# ⭐ 回避策の本体は **逐次実行**である。v1 で models_group_s45 の vit_small_r112 を
#    単独で走らせたら環境変数なしで完走した (acc=0.8875・rc=0) ことを実測で確認した。
#    UCX_HANDLE_ERRORS=none は二重の保険として足してある。
#
# ⭐ 実行ロジック・待ち方・GPU 割当・タイムアウトは v1 版 (fill_group_eval.sh) と**完全に同一**。
#    違うのは分割・出力先・ログ名だけ (正規化 diff で検証済み)。
#
# ⚠️ 本線が処理中の seed には触らない。本線は完了した seed へ戻らないので、
#    「本線が今いる seed より前」だけを対象にすれば競合しない。
#    ⭐ そのため本線の標準出力は必ず MASTER_LOG (logs/group_boundary_v2_master.log) へ
#      流すこと。run_group_v2_all.sh は tee でそこへ書いている。
# ⚠️ 条件 A の DINOv2-L は models_group_v2_s<seed>/dinov2_l_r224 が無いと評価できない
#    (境界 16 構成に N=224 を入れていないため)。追加学習は別スクリプトが行うので
#    本スクリプトは条件 A の DINOv2-L に触らない。→ train/fill_group_condA_dinov2_v2.sh
# ⚠️ 本線の summary.json を壊さないため --summary_name を分ける。
# ⚠️ GPU は 3 番だけ使う。本線の評価は 0/1/2 に割り当てられるので評価同士がぶつからない。
#
# 起動 (通常は train/run_group_v2_all.sh から呼ばれる):
#   cd /work/gfsi/ufsi0002/bs2026-resolution
#   nohup setsid bash train/fill_group_eval_v2.sh > logs/fill_group_eval_v2.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SPG=/work/gfsi/ufsi0002/bs2026-raptor/data/splits_group_v2
GPU="${FILL_GPU:-3}"
MASTER_LOG="$PROJ/logs/group_boundary_v2_master.log"
MASTER_DONE="$PROJ/logs/group_boundary_v2.done"

R50="32 48 64 80 96 112 128 160 176 208 224"
V2L="64 144 160"
VITS="112 224"
SEEDS=$(seq 42 71)

# 条件 B は 16 構成すべて、条件 A は 13 構成 (ResNet50 11 + ViT-S 2) がそろえば完了とする。
# ⚠️ 条件 A の DINOv2-L 3 構成は別スクリプトの担当なのでここでは数に入れない。
NEED_B=16
NEED_A=13

cd "$PROJ" || exit 1
[ -d "$SPG" ] || { echo "[abort] group split が無い: $SPG"; exit 1; }

APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1"
APP="$APP --env UCX_HANDLE_ERRORS=none --bind /work --bind /home1 $SIF"

# count_json <dir> : G2_cond*_s<NN> の中の構成 JSON を数える (summary.json は数えない)
count_json() {
    ls "$1" 2>/dev/null | grep -cE '^(resnet50|dinov2_l|vit_small)_r[0-9]+\.json$'
}

# 本線が今どの seed を処理しているか。完了していれば 0 を返す (どの seed も触ってよい)
current_seed() {
    [ -f "$MASTER_DONE" ] && { echo 0; return; }
    grep -oE '########## seed [0-9]+ ##########' "$MASTER_LOG" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+' | head -1
}

# eval_one <mode> <seed> <model> <resolutions>
eval_one() {
    local mode="$1" seed="$2" m="$3" rs="$4"
    local out="$PROJ/results/G2_cond${mode}_s${seed}"
    mkdir -p "$out"
    local t0; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" $APP python3 train/eval_sweep.py \
        --mode "$mode" --models "$m" --resolutions $rs \
        --models_dir "$PROJ/models_group_v2_s${seed}" --output_dir "$out" \
        --data_root "$DR" --splits_dir "$SPG" \
        --skip_existing --summary_name "summary_fill_${m}.json" \
        >> "logs/fill_ev_${mode}_${seed}_${m}.log" 2>&1
    local rc=$?
    printf "    [%s] cond%s s%-2s %-10s rc=%d %3ds -> %s/16\n" \
           "$(date '+%H:%M:%S')" "$mode" "$seed" "$m" "$rc" \
           "$(( $(date +%s)-t0 ))" "$(count_json "$out")"
    return $rc
}

T0=$(date +%s)
echo "===== group split 評価の穴埋め 開始 $(date '+%F %T') ====="
echo "  GPU ${GPU} / 逐次実行 / UCX_HANDLE_ERRORS=none"
echo "  条件 B は ${NEED_B} 構成、条件 A は ${NEED_A} 構成 (DINOv2-L 3 は別スクリプト) でそろい"

round=0
while true; do
    round=$((round+1))
    cur=$(current_seed); cur=${cur:-42}
    nfix=0
    echo "[$(date '+%F %T')] --- 巡回 ${round} (本線は seed ${cur} を処理中) ---"

    for s in $SEEDS; do
        # 本線が処理中の seed は触らない (完了後は cur=0 なので全 seed が対象になる)
        [ "$s" = "$cur" ] && continue
        MD="$PROJ/models_group_v2_s${s}"
        nm=$(ls "$MD" 2>/dev/null | wc -l)
        [ "$nm" -ge 16 ] || continue      # 学習がまだ済んでいない seed は飛ばす

        nb=$(count_json "$PROJ/results/G2_condB_s${s}")
        na=$(count_json "$PROJ/results/G2_condA_s${s}")

        if [ "$nb" -lt "$NEED_B" ]; then
            for spec in "resnet50:$R50" "dinov2_l:$V2L" "vit_small:$VITS"; do
                eval_one B "$s" "${spec%%:*}" "${spec#*:}"
            done
            nfix=$((nfix+1))
        fi
        if [ "$na" -lt "$NEED_A" ]; then
            # ⚠️ 条件 A の DINOv2-L は dinov2_l_r224 が無いので回さない (別スクリプトの担当)
            for spec in "resnet50:$R50" "vit_small:$VITS"; do
                eval_one A "$s" "${spec%%:*}" "${spec#*:}"
            done
            nfix=$((nfix+1))
        fi
    done

    # ---- 完了判定は「マーカー」ではなく**データを数えて**行う ----
    okA=0; okB=0; short=""
    for s in $SEEDS; do
        nb=$(count_json "$PROJ/results/G2_condB_s${s}")
        na=$(count_json "$PROJ/results/G2_condA_s${s}")
        [ "$nb" -ge "$NEED_B" ] && okB=$((okB+1))
        [ "$na" -ge "$NEED_A" ] && okA=$((okA+1))
        [ "$nb" -ge "$NEED_B" ] && [ "$na" -ge "$NEED_A" ] || short="$short s${s}(A${na}/B${nb})"
    done
    echo "[$(date '+%F %T')] 巡回 ${round} 終了: 今回埋めた $nfix 件 / 条件A ${okA}/30 ・条件B ${okB}/30"
    [ -n "$short" ] && echo "    未了:${short}"

    if [ "$okA" -ge 30 ] && [ "$okB" -ge 30 ]; then
        echo "===== 全 30 シードそろった $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/60 )) 分 ====="
        touch "$PROJ/logs/fill_group_eval_v2.done"
        exit 0
    fi

    # 本線が終わっているのに埋まらないものが残るなら、これ以上待っても変わらない
    if [ -f "$MASTER_DONE" ] && [ "$nfix" -eq 0 ]; then
        echo "⛔ 本線は完了しているが埋まらない構成が残った。手で調べること"
        echo "   (学習が無いのか、評価が繰り返し落ちるのかを logs/fill_ev_*.log で確認する)"
        exit 1
    fi

    # 48 時間で諦める (本線の完了見込みは 9/7 14 時ごろ)
    [ $(( ($(date +%s)-T0)/3600 )) -ge 48 ] && {
        echo "⛔ 48 時間たっても終わらないので中止する"; exit 1; }
    sleep 600
done
