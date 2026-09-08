#!/usr/bin/env bash
# ViT-L の配備エンジンで分類精度を測る (fgpu0 側・1 シード分).
#
# 査読指摘 A-4 への対応。seed 42 以外の ViT-L 配備精度が無いため、論文の図表は
# 30 シードの**サーバ FP32** 平均を代用している (CNN 側は 30 シードの配備精度)。
# 本スクリプトは 1 シード分のエンジンを作り、全数 1,882 枚を流して CSV を残す。
#
# ⭐ 演算精度は seed 42 の配備構成をそのまま踏襲する (変えると別物の測定になる):
#   DINOv2-L 5 段: p0/p1/p2/p3s0 は FP16、**p3s1 のみ FP32** (活性 3.6e5 が FP16 上限超過)
#   DINOv3-L 4 段: **全段 FP32** (段境界の残差が 1.55e5 で全段超過)
# ⚠️ FP32 にする唯一確実な手段は `--fp16` を付けないこと。TRT 10.3 では
#    `--precisionConstraints=obey --layerPrecisions=*:fp32` が警告なく無視される。
#
# ⚠️ レイテンシは測らない。グラフはシードによらず同一で、変わるのは重みだけだからである。
#    レイテンシは既存の 30 回測定 (seed 42) を使う。ここで要るのは task accuracy だけ。
#
# usage:
#   bash run_vitl_acc_seed.sh 43            # 全 14 水準
#   bash run_vitl_acc_seed.sh 43 112        # 1 水準だけ (パイロット)
#   KEEP=1 bash run_vitl_acc_seed.sh 43     # エンジンと ONNX を消さない
set -u

SEED="${1:?usage: run_vitl_acc_seed.sh <seed> [res...]}"; shift
RES="${*:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
S="$E/vitl_seeds/s${SEED}"
ENG="$S/engines"
OUT="$E/preds_vitl_30seed/s${SEED}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
CHAIN=./orin_infer_chain_trt10
NTEST=1882

[ -x "$CHAIN" ] || { echo "[abort] $CHAIN が無い"; exit 1; }
[ -d "$S" ] || { echo "[abort] ONNX が無い: $S (HPC 側の prep_vitl_seed.sh を先に)"; exit 1; }
mkdir -p "$ENG" "$OUT" logs/vitl_s${SEED}

# ---- 競合チェック。⚠️ pgrep -f は自分のコマンドラインにも一致するので使わない ----
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then
    echo "[abort] trtexec が ${n} 本走っている。測定と競合するので中止する"; exit 1
fi

# build <onnx> <engine> <prec>   prec は "--fp16" か "" (空なら FP32)
nb=0; nsk=0; nf=0; nm=0
build() {
    local onnx="$1" eng="$2" prec="$3"
    local name; name=$(basename "$eng" .engine)
    [ -s "$onnx" ] || { echo "    [MISSING] $name <- $onnx"; nm=$((nm+1)); return 1; }
    [ -s "$eng" ] && { nsk=$((nsk+1)); return 0; }
    local t0; t0=$(date +%s)
    # ⭐ $prec はクォートしない (空なら引数ごと消える)
    if timeout 3600 "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" $prec \
            > "logs/vitl_s${SEED}/b_${name}.log" 2>&1; then
        printf "    [OK] %-28s %3ds %5d MB %s\n" "$name" "$(( $(date +%s)-t0 ))" \
               "$(( $(stat -c %s "$eng") / 1000000 ))" "${prec:-FP32}"
        nb=$((nb+1)); return 0
    fi
    echo "    [FAILED] $name"; grep -aE "\[E\]|Killed|out of memory" "logs/vitl_s${SEED}/b_${name}.log" | tail -2
    rm -f "$eng"; nf=$((nf+1)); return 1
}

# distinct <csv> : ヘッダを除いた相異なる出力の数
distinct() { tail -n +2 "$1" 2>/dev/null | cut -d, -f2- | sort -u | wc -l; }

n_ok=0; n_ng=0; n_skip=0; n_const=0
run_chain() {
    local csv="$1" bin="$2" label="$3"; shift 3
    local e
    for e in "$@"; do [ -s "$e" ] || { echo "    [miss] $label"; n_ng=$((n_ng+1)); return; }; done
    [ -s "$bin" ] || { echo "    [miss] $label (入力 bin 無)"; n_ng=$((n_ng+1)); return; }
    if [ -s "$csv" ] && [ "$(wc -l < "$csv")" -eq $((NTEST+1)) ]; then
        n_skip=$((n_skip+1)); return; fi
    local t0; t0=$(date +%s)
    "$CHAIN" "$csv" "$NTEST" "$bin" "$@" > "logs/vitl_s${SEED}/i_${label}.log" 2>&1
    local rc=$? d; d=$(distinct "$csv")
    printf "    %-24s rc=%d %3ds 相異なる出力=%s/%d%s\n" "$label" "$rc" \
           "$(( $(date +%s)-t0 ))" "$d" "$NTEST" \
           "$([ "$d" -le 1 ] && echo '  << 定数 (無効)')"
    if [ "$rc" -ne 0 ]; then n_ng=$((n_ng+1)); else n_ok=$((n_ok+1)); fi
    # ⚠️ 定数出力の判定は「相異なる出力 <= 1」では甘い (3/1882 が素通りした前例あり)。
    #    正常なら logits は連続値なので n/n になる。5% を境にする。
    [ "$d" -lt $((NTEST/20)) ] && { n_const=$((n_const+1)); echo "      ⚠ 相異なる出力が 5% 未満。無効とみなす"; }
}

T0=$(date +%s)
echo "===== ViT-L 配備精度 seed ${SEED} 開始 $(date '+%F %T') ====="
echo "  水準: $RES"

for R in $RES; do
    echo "  [$(date '+%H:%M:%S')] N=${R}"
    # -------- DINOv2-L (5 段・最終段のみ FP32) --------
    v2ok=1
    for p in p0 p1 p2; do
        build "$S/onnx_split_raw/dinov2_l_r${R}_fp16_${p}.onnx" \
              "$ENG/dinov2_l_r${R}_${p}.engine" "--fp16" || v2ok=0
    done
    build "$S/onnx_split_raw_p3/dinov2_l_r${R}_fp16_p3_p0.onnx" \
          "$ENG/dinov2_l_r${R}_p3s0.engine" "--fp16" || v2ok=0
    build "$S/onnx_split_raw_p3/dinov2_l_r${R}_fp16_p3_p1.onnx" \
          "$ENG/dinov2_l_r${R}_p3s1.engine" "" || v2ok=0
    [ "$v2ok" = "1" ] && run_chain "$OUT/dinov2_l_r${R}.csv" "inputs/real_dinov2_l_r${R}.fp16.bin" \
        "dinov2_l_r${R}" "$ENG"/dinov2_l_r${R}_{p0,p1,p2,p3s0,p3s1}.engine

    # -------- DINOv3-L (4 段・全段 FP32) --------
    v3ok=1
    for k in 0 1 2 3; do
        build "$S/onnx_split/dinov3_l_r${R}_fp16_sim_p${k}_ort.onnx" \
              "$ENG/dinov3_l_r${R}_p${k}.engine" "" || v3ok=0
    done
    [ "$v3ok" = "1" ] && run_chain "$OUT/dinov3_l_r${R}.csv" "inputs/real_dinov3_l_r${R}.fp16.bin" \
        "dinov3_l_r${R}" "$ENG"/dinov3_l_r${R}_p{0,1,2,3}.engine

    # 水準ごとにエンジンを捨てる (1 水準で約 4 GB。全部残すと 1 シード 56 GB になる)
    if [ "${KEEP:-0}" != "1" ]; then
        rm -f "$ENG"/dinov2_l_r${R}_*.engine "$ENG"/dinov3_l_r${R}_*.engine
    fi
done

echo "===== seed ${SEED} 完了 $(date '+%F %T') ====="
echo "  ビルド ${nb} / スキップ ${nsk} / 失敗 ${nf} / ONNX 欠品 ${nm}"
echo "  推論 成功 ${n_ok} / スキップ ${n_skip} / 失敗 ${n_ng} / 定数 ${n_const}"
echo "  CSV $(ls "$OUT"/*.csv 2>/dev/null | wc -l) 件 / 所要 $(( ($(date +%s)-T0)/60 )) 分"

# ⚠️⚠️ マーカーは「失敗 0」ではなく「**全 28 構成の CSV がそろっている**」ことで立てる。
#    2026-09-05 に実際に踏んだ: パイロットで 1 水準だけ流したらマーカーが立ち、
#    チェーンがそのシードを「測定済み」として飛ばした。
FULL="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
have=0; miss=""
for R in $FULL; do
    for M in dinov2_l dinov3_l; do
        c="$OUT/${M}_r${R}.csv"
        if [ -s "$c" ] && [ "$(wc -l < "$c")" -eq $((NTEST+1)) ]; then
            have=$((have+1))
        else
            miss="$miss ${M}_r${R}"
        fi
    done
done
echo "  そろった構成: ${have} / 28"

if [ "$nf" -eq 0 ] && [ "$n_ng" -eq 0 ] && [ "$n_const" -eq 0 ] && [ "$have" -eq 28 ]; then
    touch "$E/logs/vitl_acc_s${SEED}.done"
    # ONNX も落とす (1 シード 27 GB)。CSV は残る
    [ "${KEEP:-0}" != "1" ] && rm -rf "$S/onnx_split_raw" "$S/onnx_split_raw_p3" "$S/onnx_split" "$S/onnx"
    exit 0
fi
rm -f "$E/logs/vitl_acc_s${SEED}.done"
echo "  ⚠ マーカーを立てない (ビルド失敗 ${nf} / 推論失敗 ${n_ng} / 定数 ${n_const} / 未了:${miss:- なし})"
echo "     ONNX は残すので、同じコマンドで再開できる"
exit 1
