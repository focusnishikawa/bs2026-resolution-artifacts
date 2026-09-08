#!/usr/bin/env bash
# 蒸留の低解像度耐性 ablation (PLAN.md「Phase 5 で ablation として言及する」の宿題).
#
# 動機: 既存 raptor の multires 実験 (蒸留済み生徒 student_effb0_T2a05_v3) は
#       effb0: 32→0.636 / 64→0.807 / 128→0.905 / 224→0.928 と、本 PJ の CE-only を
#       低解像度ほど大きく上回っていた (N=32 で 0.636 対 0.538)。
#       ただし蒸留の有無以外にデータ・レシピ・seed が揃っていないため、そのままでは
#       「蒸留が低解像度耐性を高める」根拠にならない。ここで統制して確かめる。
#
# 統制: データ・split・レシピ・学習上限・3 seed をすべて条件 B の CE-only と揃え、
#       **違いは損失だけ**にする。
#   生徒 = EfficientNet-B0 @ N=224 (条件 B の N=224 と同じ設定)
#   教師 = 本 PJ Phase 1 の DINOv2-L @224 (test acc 0.9772)
#   損失 = 0.5 * T^2 * KL(student/T || teacher/T) + 0.5 * CE、T=2 (raptor の T2a05 に合わせた)
#
# 学習後に 14 解像度の劣化耐性 (条件 A) を評価し、既存 CE-only 3 seed と比べる。
#
# ⚠️ GPU は 2,3 のみ使う。GPU0-1 は他セッション (HPC207 の vLLM) が占有しており、
#    先方の記載「GPU2-3 は最後まで一切使わない / vLLM 以外なら可」に従う。
#
# 起動: nohup setsid bash train/run_kd_ablation.sh > logs/kd_master.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
DR=/work/gfsi/ufsi0002/bs2026-raptor/data_root
SP=/work/gfsi/ufsi0002/bs2026-raptor/data/splits
PW=/work/gfsi/ufsi0002/bs2026-raptor/weights
TEACHER_CKPT=${PROJ}/models_s42/dinov2_l_r224/best.pt
RES=224
SEEDS="42 43 44"
GPUS="2 3"          # GPU0-1 は他セッションが占有中。触らない

APP="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --bind /work --bind /home1 $SIF"
cd "$PROJ" || exit 1
mkdir -p logs models_kd

[ -f "$TEACHER_CKPT" ] || { echo "[NG] 教師が無い: $TEACHER_CKPT"; exit 1; }

echo "[$(date '+%H:%M:%S')] ===== 蒸留 ablation 開始 (effb0 @ ${RES}, 3 seed) ====="
i=0
for seed in $SEEDS; do
    gpu=$(echo $GPUS | cut -d' ' -f$(( i % 2 + 1 )))
    tag="effb0_r${RES}_kd_s${seed}"
    out="${PROJ}/models_kd/${tag}"
    log="${PROJ}/logs/kd_${tag}.log"
    rm -f "${PROJ}/logs/kd_${tag}.done"
    mkdir -p "$out"
    echo "  launch gpu=${gpu} ${tag}"
    nohup setsid bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} ${APP} python3 train/train_res.py \
            --model effb0 --res ${RES} --mode B --seed ${seed} --num_workers 8 \
            --data_root ${DR} --splits_dir ${SP} --pretrain_dir ${PW} \
            --teacher_model dinov2_l --teacher_ckpt ${TEACHER_CKPT} \
            --kd_alpha 0.5 --kd_T 2.0 \
            --output ${out} > ${log} 2>&1
        echo DONE_RC=\$? >> ${log}
        touch ${PROJ}/logs/kd_${tag}.done
    " > /dev/null 2>&1 < /dev/null &
    i=$((i + 1))
    sleep 20
done

# 3 本目は 2 GPU に 2 本目までを流したあと同じ GPU に相乗りするので、
# 進捗は .done ファイルで待つ
echo "[$(date '+%H:%M:%S')] 3 本投入。完了を待つ"
n=0
while true; do
    done_n=$(ls ${PROJ}/logs/kd_effb0_r${RES}_kd_s*.done 2>/dev/null | wc -l)
    [ "$done_n" -ge 3 ] && break
    sleep 60
    n=$((n + 1))
    if [ $((n % 10)) -eq 0 ]; then
        echo "  [$(date '+%H:%M:%S')] ${done_n}/3 完了"
    fi
done

echo "[$(date '+%H:%M:%S')] ===== 学習完了。test 精度 ====="
for seed in $SEEDS; do
    f="${PROJ}/models_kd/effb0_r${RES}_kd_s${seed}/test_metrics.json"
    [ -f "$f" ] && echo "  seed ${seed}: $(python3 -c "import json;d=json.load(open('$f'));print('acc=%.4f mR=%.4f'%(d['acc'],d.get('macro_recall',0)))" 2>/dev/null)"
done
touch "${PROJ}/logs/kd_train.done"
