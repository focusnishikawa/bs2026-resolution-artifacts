#!/usr/bin/env bash
# ViT-L の ONNX 書き出しチェーン (**m001 版**).
#
# hpciaiss1 側の stage_vitl_ckpt_chain.sh が立てる .stage.done を待ってから 1 シードずつ
# 書き出す。出力先は /home1 の共有 FS なので、fgpu0 側のチェーンが export.done を見て
# エンジン構築へ進む。
#
# ⛔ hpciaiss1 では計算しない (複製のみ)。書き出しは m001、エンジンと推論は fgpu0。
# ⚠️ ckpt は書き出しが済み次第 hpciaiss1 側チェーンが消すので、**先に export.done を立てる**
#    順序を崩さないこと (本スクリプトは prep_vitl_seed_m001.sh の終了後に何も消さない)。
#
# usage:
#   nohup setsid bash train/run_vitl_export_chain_m001.sh > logs/vitl_export_chain_m001.log 2>&1 &
set -u
PROJ="${PROJ:-/home1/gfsi/ufsi0002/bs2026-resolution-code}"
STAGE="${STAGE:-/home1/gfsi/ufsi0002/bs2026-resolution-ckpt}"
E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SEEDS=$(seq 43 71)
MAXROUND="${MAXROUND:-4}"

cd "$PROJ" || exit 1
mkdir -p logs
T0=$(date +%s)
echo "===== ViT-L ONNX 書き出しチェーン (m001) 開始 $(date '+%F %T') ====="
echo "  ホスト: $(hostname) / コード: $PROJ / ckpt: $STAGE"

for round in $(seq 1 "$MAXROUND"); do
    n=$(ls "$E"/vitl_seeds/*/export.done 2>/dev/null | wc -l)
    echo "########## 巡回 ${round}/${MAXROUND} (完了 ${n}/29) $(date '+%F %T') ##########"
    [ "$n" -ge 29 ] && break
    for s in $SEEDS; do
        [ -f "$E/vitl_seeds/s${s}/export.done" ] && continue
        # ckpt の複製を待つ (最大 8 時間)
        w=0
        while [ ! -f "$STAGE/models_s${s}/.stage.done" ]; do
            [ "$w" -ge 480 ] && { echo "[abort] seed $s の ckpt 複製を 8 時間待ったが来ない"; exit 1; }
            [ $((w % 30)) -eq 0 ] && echo "[wait] seed $s の ckpt 待ち ($w 分)"
            sleep 60; w=$((w+1))
        done
        echo "########## seed $s 書き出し開始 $(date '+%F %T') ##########"
        bash train/prep_vitl_seed_m001.sh "$s"
        echo "########## seed $s 書き出し終了 rc=$? $(date '+%F %T') ##########"
    done
done

n=$(ls "$E"/vitl_seeds/*/export.done 2>/dev/null | wc -l)
echo "===== 終了 $(date '+%F %T') 完了 ${n}/29 ・所要 $(( ($(date +%s)-T0)/3600 )) 時間 ====="
[ "$n" -ge 29 ] && touch logs/vitl_export_chain_m001.done
exit 0
