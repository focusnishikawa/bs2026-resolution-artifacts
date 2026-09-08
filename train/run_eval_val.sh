#!/usr/bin/env bash
# 査読指摘 A5「テスト集合を構成選択に使っている」への対応:
#   84 構成のパレートフロント・精度閾値・N の選択を test 集合で行い、同じ集合から
#   最終性能を報告すると選択バイアスが入る。そこで **validation で選択し、test で
#   最終評価する**ように分ける。ここでは前段として **条件 B の validation 予測**を作る。
#
# 条件 A は §4 の記述統計にのみ使い構成選択には用いないので対象外。
#
# ⚠️ N=16 の扱いは run_eval_v2.sh の実測に従う:
#   条件 B の N=16 は入力 16x16 (patch16 ViT でトークン 1 個) で **CUDA 上で SIGSEGV** する。
#   -> N=16 だけ CPU で実行する。入力が小さいので CPU でも数十秒。
#
# usage: bash train/run_eval_val.sh [開始seed] [終了seed]
# 起動:  nohup setsid bash train/run_eval_val.sh 42 71 > logs/eval_val_master.log 2>&1 < /dev/null &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP=/home1/gfsi/ufsi0002/apptainer/bin/apptainer
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
# UCX_HANDLE_ERRORS=none は UCX のシグナルハンドラを外すための指定。
# これが無いと eval_sweep.py が UCX の spinlock 破棄で SIGSEGV (rc=139) を起こし、
# 2026-09-02 の試験では effb0 と dinov3_l の 25 構成が失われた (backtrace は libucs.so)。
UCXENV="--env UCX_HANDLE_ERRORS=none --env UCX_LOG_LEVEL=error"
APP_BASE="$APP exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 $UCXENV --bind /work --bind /home1 $SIF"
APP_CPU="$APP exec --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 $UCXENV --bind /work --bind /home1 $SIF"

RES_NO16="32 48 64 80 96 112 128 144 160 176 192 208 224"
S0="${1:-42}"; S1="${2:-71}"

cd "$PROJ" || exit 1
mkdir -p logs results
# 前回の .done が残っていると「完了」と誤読するので必ず消す
rm -f logs/eval_val.done
echo "===== validation 予測 (条件 B) seed ${S0}-${S1} 開始 $(date '+%F %T') ====="

for seed in $(seq "$S0" "$S1"); do
    MD="${PROJ}/models_s${seed}"
    OUT="${PROJ}/results/T1_condB_s${seed}_val"
    if [ ! -d "$MD" ]; then echo "[miss] models_s${seed} が無い"; continue; fi
    # 84 構成そろっていれば作り直さない (中断しても同じコマンドで再開できる)
    if [ "$(ls "$OUT"/preds/*.npz 2>/dev/null | wc -l)" -ge 84 ]; then
        echo "[skip] seed ${seed} (84 構成そろい)"; continue
    fi
    mkdir -p "$OUT"
    T0=$(date +%s)
    echo "[$(date '+%H:%M:%S')] ===== seed ${seed} 開始 ====="

  # UCX の SIGSEGV で 1 回の実行では 25 構成ほど落ちることがある (effb0 と dinov3_l)。
  # --skip_existing があるので、そろうまで繰り返せば埋まる。実測では 2 周でそろう。
  for attempt in 1 2 3; do
    n_have=$(ls "$OUT"/preds/*.npz 2>/dev/null | wc -l)
    [ "$n_have" -ge 84 ] && break
    [ "$attempt" -gt 1 ] && echo "  [retry ${attempt}] seed ${seed} (${n_have}/84)"
    rm -f logs/evval_${seed}_gpu*.done

    i=0
    for models in "mnv4 dinov2_l" "effb0 dinov3_l" "resnet50" "vit_small"; do
        tag="evval_${seed}_gpu${i}"
        nohup setsid bash -c "
            CUDA_VISIBLE_DEVICES=${i} ${APP_BASE} python3 train/eval_sweep.py \
                --mode B --split val --models ${models} --resolutions ${RES_NO16} \
                --models_dir ${MD} --output_dir ${OUT} \
                --data_root ${DR} --splits_dir ${SP} \
                --skip_existing --torch_threads 8 --summary_name summary_val_gpu${i}.json \
                > ${PROJ}/logs/${tag}.log 2>&1
            echo DONE_RC=\$? >> ${PROJ}/logs/${tag}.log
            touch ${PROJ}/logs/${tag}.done
        " > /dev/null 2>&1 < /dev/null &
        i=$((i + 1))
        sleep 2
    done

    while [ "$(ls logs/evval_${seed}_gpu*.done 2>/dev/null | wc -l)" -lt 4 ]; do
        sleep 20
    done

    # N=16 は CPU で (CUDA だと SIGSEGV する)
    CUDA_VISIBLE_DEVICES= ${APP_CPU} python3 train/eval_sweep.py \
        --mode B --split val --resolutions 16 \
        --models_dir "${MD}" --output_dir "${OUT}" \
        --data_root "${DR}" --splits_dir "${SP}" \
        --skip_existing --torch_threads 16 --summary_name summary_val_cpu.json \
        > "${PROJ}/logs/evval_${seed}_cpu.log" 2>&1
  done

    n=$(ls "$OUT"/preds/*.npz 2>/dev/null | wc -l)
    printf "[%s] seed %s 完了: %d/84 構成  %d 分\n" "$(date '+%H:%M:%S')" "$seed" "$n" \
           "$(( ($(date +%s) - T0) / 60 ))"
done

echo "===== 完了 $(date '+%F %T') ====="
n_short=0
for seed in $(seq "$S0" "$S1"); do
    d="${PROJ}/results/T1_condB_s${seed}_val/preds"
    n=$(ls "$d"/*.npz 2>/dev/null | wc -l)
    [ "$n" -lt 84 ] && n_short=$((n_short+1))
done
[ "$n_short" -gt 0 ] && echo "⚠️ 84 構成に満たない seed が ${n_short} 件ある。もう一度流すこと"
echo "seed 別の構成数:"
for seed in $(seq "$S0" "$S1"); do
    d="${PROJ}/results/T1_condB_s${seed}_val/preds"
    [ -d "$d" ] && printf "  s%-3s %d/84\n" "$seed" "$(ls "$d"/*.npz 2>/dev/null | wc -l)"
done
touch logs/eval_val.done
exit 0
