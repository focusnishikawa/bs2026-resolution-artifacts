#!/usr/bin/env bash
# ViT-L チェックポイントを hpciaiss1 の /work から **共有 FS の /home1 へ複製する**チェーン.
#
# 2026-09-08 の再レビュー指摘 A-5 (ViT-L の validation を配備エンジンで実測する) のための下準備。
#
# ⛔⛔ **hpciaiss1 は他セッションが計算に使っているので、ここでは計算を一切しない。**
#     本スクリプトがするのは `cp` だけである (ONNX 書き出しは m001、エンジン構築と推論は fgpu0)。
# ⚠️ m001 の /work は**ローカル NVMe** で hpciaiss1 の /work とは別物であり、
#    hpciaiss1 へ SSH も張れない。**共有 FS は /home1 (Lustre) だけ**なので、そこを経由する。
#
# 1 シードあたり ViT-L の ckpt は 28 個 × 1.2 GB = **約 34 GB**、29 シードで約 970 GB。
# 全部を置きっぱなしにしないよう、**書き出しが済んだシードから消す** (MAXAHEAD 段先まで)。
#
# usage:
#   nohup setsid bash train/stage_vitl_ckpt_chain.sh > logs/stage_ckpt_chain.log 2>&1 &
set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
STAGE=/home1/gfsi/ufsi0002/bs2026-resolution-ckpt
EDGE=/home1/gfsi/ufsi0002/bs2026-resolution-edge
SEEDS=$(seq 43 71)
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
MAXAHEAD="${MAXAHEAD:-3}"      # 書き出し待ちの在庫をこの数までにする (3 × 34 GB = 102 GB)

cd "$PROJ" || exit 1
mkdir -p "$STAGE" logs
T0=$(date +%s)
echo "===== ViT-L ckpt 複製チェーン 開始 $(date '+%F %T') ====="
echo "  複製元: $PROJ/models_s<NN>  ->  $STAGE/models_s<NN>"
echo "  在庫上限: ${MAXAHEAD} シード"

# 書き出しが済んだシードの ckpt を消す
cleanup() {
    local s
    for s in $SEEDS; do
        if [ -f "$EDGE/vitl_seeds/s${s}/export.done" ] && [ -d "$STAGE/models_s${s}" ]; then
            echo "  [clean] s${s} の ckpt を削除 ($(du -sh "$STAGE/models_s${s}" 2>/dev/null | cut -f1))"
            rm -rf "$STAGE/models_s${s}"
        fi
    done
}

# 在庫 = 複製済みだが書き出しがまだのシード数
stock() {
    local n=0 s
    for s in $SEEDS; do
        [ -f "$STAGE/models_s${s}/.stage.done" ] && [ ! -f "$EDGE/vitl_seeds/s${s}/export.done" ] \
            && n=$((n+1))
    done
    echo "$n"
}

for s in $SEEDS; do
    if [ -f "$STAGE/models_s${s}/.stage.done" ] || [ -f "$EDGE/vitl_seeds/s${s}/export.done" ]; then
        echo "[skip] seed $s (複製済みまたは書き出し済み)"; continue
    fi
    # 在庫が上限に達している間は待つ
    w=0
    while [ "$(stock)" -ge "$MAXAHEAD" ]; do
        cleanup
        [ "$(stock)" -lt "$MAXAHEAD" ] && break
        [ $((w % 30)) -eq 0 ] && echo "[wait] 在庫 $(stock)/${MAXAHEAD} のため待機 ($w 分)"
        sleep 60; w=$((w+1))
        [ "$w" -ge 720 ] && { echo "[abort] 12 時間待っても在庫が減らない"; exit 1; }
    done

    t0=$(date +%s)
    dst="$STAGE/models_s${s}"
    mkdir -p "$dst"
    ng=0
    for R in $RES; do
        for M in dinov2_l dinov3_l; do
            src="$PROJ/models_s${s}/${M}_r${R}"
            [ -s "$src/best.pt" ] || { echo "  [MISSING] s${s} ${M}_r${R}"; ng=$((ng+1)); continue; }
            [ -s "$dst/${M}_r${R}/best.pt" ] && continue
            mkdir -p "$dst/${M}_r${R}"
            cp -p "$src/best.pt" "$dst/${M}_r${R}/best.pt" || ng=$((ng+1))
            [ -s "$src/meta.json" ] && cp -p "$src/meta.json" "$dst/${M}_r${R}/" 2>/dev/null
        done
    done
    # ⚠️ マーカーは「失敗 0」ではなく **28 個そろったこと**で立てる
    have=$(ls "$dst"/*/best.pt 2>/dev/null | wc -l)
    printf "[%s] seed %d 複製 %d/28 ・%s ・%d 分\n" "$(date '+%H:%M:%S')" "$s" "$have" \
           "$(du -sh "$dst" 2>/dev/null | cut -f1)" "$(( ($(date +%s)-t0)/60 ))"
    if [ "$have" -eq 28 ]; then
        touch "$dst/.stage.done"
    else
        echo "  ⚠ マーカーを立てない (欠品 $((28-have)) 件 / cp 失敗 ${ng} 件)"
    fi
done

echo "===== 複製チェーン終了 $(date '+%F %T') 所要 $(( ($(date +%s)-T0)/60 )) 分 ====="
touch "$STAGE/stage_chain.done"
# 後始末は書き出しが進むにつれて必要になるので、しばらく監視して消す
for i in $(seq 1 720); do
    cleanup
    n=$(ls -d "$STAGE"/models_s* 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && { echo "全 ckpt を削除した $(date '+%F %T')"; break; }
    sleep 300
done
