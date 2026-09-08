#!/usr/bin/env bash
# TensorRT 10.3 エンジンで全数推論をやり直す (argmax 一致率・精度を新環境へ揃える).
#
# 論文の Orin 値を全て TRT 10.3 に差し替える方針 (ユーザー指示 2026-08-31) なので、
# latency だけでなく **エンジンの出力**も取り直す。
#
# ⚠️ 「入力依存性」を必ず確認すること。FP16 の飽和で壊れたエンジンは入力によらず
#    同じ値を返すようになり、しかも計算が消えるぶん**正常なものより速く見える**。
#    latency だけでは検出できないので、相異なる出力の本数をここで数える。
#
# 対象 (入力 bin がある範囲に限られる):
#   CNN/ViT-S  : 4 モデル x 14 解像度 = 56   <- 2026-09-02 に入力 bin を全 14 水準そろえた
#   ViT-S FP32 : 2 (r112/r224)                           <- Orin FP32 対サーバの対照
#   ViT-L 配備 : DINOv2-L 14 + DINOv3-L 14 = 28 チェーン  <- real_*_r*.fp16.bin は 14 水準揃っている
#   ViT-L 対照 : DINOv2-L 全段 FP32 6 チェーン
#   ViT-L 参照 : DINOv3-L 単体エンジン 1
#
# usage: bash infer_trt10_all.sh [smoke|all]
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
ENGDIR=engines_trt10_split
OUTDIR=preds_trt10
NTEST=1882
CHAIN=./orin_infer_chain_trt10
RES1=./orin_infer_res_trt10
# ⭐ 2026-09-02: 入力 bin を全 14 水準そろえたので既定を 14 水準へ広げた
#    (それまでは inputs_r{16,112,224}.bin しか無く 3 水準に限られていた)。
#    査読指摘 A1「配備精度をサーバ精度で代用している」に答えるには、
#    **配備するエンジンそのもので全構成の task accuracy を測る**必要がある。
#    ⚠️ run_res は完全な CSV があればスキップするので、既存 12 構成は再実行されない。
CNN_RES="${CNN_RES:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"
CNN_MODELS="mnv4 effb0 resnet50 vit_small"
VITL_RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
CTRL_RES="16 32 64 112 128 224"

WHAT="${1:-all}"
mkdir -p "$OUTDIR" logs
[ "$WHAT" = "all" ] && rm -f logs/infer_trt10.done

for b in "$CHAIN" "$RES1"; do
    [ -x "$b" ] || { echo "[abort] $b が無い。先にコンパイルすること"; exit 1; }
done

# distinct <csv> : ヘッダを除いて相異なる出力が何通りあるか
distinct() { tail -n +2 "$1" 2>/dev/null | cut -d, -f2- | sort -u | wc -l; }

# ---- スモークテスト (移植したバイナリが実際に動くか) ----
# コンパイルが通っても TRT10 の enqueueV3 経路が動く保証はないので、8 枚だけ流して確かめる
echo "===== スモークテスト ====="
smoke_ok=1
if [ -s engines_trt10/effb0_r112.engine ] && [ -s inputs/inputs_r112.bin ]; then
    "$RES1" engines_trt10/effb0_r112.engine inputs/inputs_r112.bin 8 112 \
        "$OUTDIR/_smoke_res.csv" > logs/smoke_res.log 2>&1
    rc=$?; d=$(distinct "$OUTDIR/_smoke_res.csv")
    echo "  orin_infer_res_trt10 : rc=$rc 行=$(wc -l < "$OUTDIR/_smoke_res.csv") 相異なる出力=${d}/8"
    { [ $rc -ne 0 ] || [ "$d" -le 1 ]; } && smoke_ok=0
else
    echo "  [skip] CNN エンジンか入力が無い"; smoke_ok=0
fi
if [ -s "$ENGDIR/dinov2_l_r32_p0.engine" ]; then
    "$CHAIN" "$OUTDIR/_smoke_chain.csv" 8 inputs/real_dinov2_l_r32.fp16.bin \
        "$ENGDIR"/dinov2_l_r32_{p0,p1,p2,p3s0,p3s1}.engine > logs/smoke_chain.log 2>&1
    rc=$?; d=$(distinct "$OUTDIR/_smoke_chain.csv")
    echo "  orin_infer_chain_trt10: rc=$rc 行=$(wc -l < "$OUTDIR/_smoke_chain.csv") 相異なる出力=${d}/8"
    { [ $rc -ne 0 ] || [ "$d" -le 1 ]; } && smoke_ok=0
else
    echo "  [skip] ViT-L エンジンがまだ無い (ビルド前ならこれで正常)"
fi
if [ "$smoke_ok" -eq 1 ]; then echo "  => スモーク OK"; else echo "  => ⚠️ スモークに問題あり。ログを見ること"; fi
[ "$WHAT" = "smoke" ] && exit 0

n_ok=0; n_skip=0; n_ng=0; n_const=0

# run_res <engine> <bin> <res> <out.csv> <label>
run_res() {
    local eng="$1" bin="$2" res="$3" csv="$4" label="$5"
    if [ ! -s "$eng" ]; then echo "  [miss] $label (エンジン無)"; n_ng=$((n_ng+1)); return; fi
    if [ ! -s "$bin" ]; then echo "  [miss] $label (入力無)";   n_ng=$((n_ng+1)); return; fi
    if [ -s "$csv" ] && [ "$(wc -l < "$csv")" -eq $((NTEST+1)) ]; then
        echo "  [skip] $label"; n_skip=$((n_skip+1)); return; fi
    local t0; t0=$(date +%s)
    "$RES1" "$eng" "$bin" "$NTEST" "$res" "$csv" > "logs/infer_${label}.log" 2>&1
    local rc=$? d; d=$(distinct "$csv")
    printf "  %-28s rc=%d %4ds 相異なる出力=%s/%d%s\n" "$label" "$rc" \
           "$(( $(date +%s) - t0 ))" "$d" "$NTEST" \
           "$([ "$d" -le 1 ] && echo '  << 定数 (無効)')"
    if [ $rc -ne 0 ]; then n_ng=$((n_ng+1)); else n_ok=$((n_ok+1)); fi
    [ "$d" -le 1 ] && n_const=$((n_const+1))
}

# run_chain <out.csv> <bin> <label> <engine...>
run_chain() {
    local csv="$1" bin="$2" label="$3"; shift 3
    for e in "$@"; do
        [ -s "$e" ] || { echo "  [miss] $label ($(basename "$e") 無)"; n_ng=$((n_ng+1)); return; }
    done
    [ -s "$bin" ] || { echo "  [miss] $label (入力無)"; n_ng=$((n_ng+1)); return; }
    if [ -s "$csv" ] && [ "$(wc -l < "$csv")" -eq $((NTEST+1)) ]; then
        echo "  [skip] $label"; n_skip=$((n_skip+1)); return; fi
    local t0; t0=$(date +%s)
    "$CHAIN" "$csv" "$NTEST" "$bin" "$@" > "logs/infer_${label}.log" 2>&1
    local rc=$? d; d=$(distinct "$csv")
    printf "  %-28s rc=%d %4ds 相異なる出力=%s/%d%s\n" "$label" "$rc" \
           "$(( $(date +%s) - t0 ))" "$d" "$NTEST" \
           "$([ "$d" -le 1 ] && echo '  << 定数 (無効)')"
    if [ $rc -ne 0 ]; then n_ng=$((n_ng+1)); else n_ok=$((n_ok+1)); fi
    [ "$d" -le 1 ] && n_const=$((n_const+1))
}

T0=$(date +%s)
echo
echo "===== CNN / ViT-S ($(echo $CNN_MODELS | wc -w) モデル x $(echo $CNN_RES | wc -w) 解像度) ====="
for m in $CNN_MODELS; do
    for r in $CNN_RES; do
        run_res "engines_trt10/${m}_r${r}.engine" "inputs/inputs_r${r}.bin" "$r" \
                "$OUTDIR/${m}_r${r}.csv" "${m}_r${r}"
    done
done

echo
echo "===== ViT-S/16 の FP32 対照 (2 構成) ====="
for r in 112 224; do
    run_res "$ENGDIR/vit_small_r${r}_fp32.engine" "inputs/inputs_r${r}.bin" "$r" \
            "$OUTDIR/vit_small_r${r}_FP32.csv" "vit_small_r${r}_FP32"
done

echo
echo "===== ViT-L 配備構成 (28 チェーン) ====="
for r in $VITL_RES; do
    run_chain "$OUTDIR/dinov2_l_r${r}_full.csv" "inputs/real_dinov2_l_r${r}.fp16.bin" \
              "dinov2_l_r${r}" "$ENGDIR"/dinov2_l_r${r}_{p0,p1,p2,p3s0,p3s1}.engine
    run_chain "$OUTDIR/dinov3_l_r${r}_full.csv" "inputs/real_dinov3_l_r${r}.fp16.bin" \
              "dinov3_l_r${r}" "$ENGDIR"/dinov3_l_r${r}_p{0,1,2,3}.engine
done

echo
echo "===== ViT-L 対照: DINOv2-L 全段 FP32 (6 チェーン) ====="
for r in $CTRL_RES; do
    run_chain "$OUTDIR/dinov2_l_r${r}_f32_full.csv" "inputs/real_dinov2_l_r${r}.fp16.bin" \
              "dinov2_l_r${r}_f32" "$ENGDIR"/dinov2_l_r${r}_f32_p{0,1,2,3}.engine
done

echo
echo "===== ViT-L 参照: 単体エンジン ====="
for tag in dinov2_l dinov3_l; do
    run_chain "$OUTDIR/${tag}_r32_single.csv" "inputs/real_${tag}_r32.fp16.bin" \
              "${tag}_r32_single" "$ENGDIR/${tag}_r32_single.engine"
done

T1=$(date +%s)
echo
echo "===== 完了 $(date '+%F %T') ====="
echo "成功 ${n_ok} / スキップ ${n_skip} / 失敗 ${n_ng} / ⚠️定数出力 ${n_const}"
echo "所要 $(( (T1-T0) / 60 )) 分"
echo "CSV: $(ls "$OUTDIR"/*.csv 2>/dev/null | wc -l) 件"
[ "$n_const" -gt 0 ] && echo "⛔ 定数出力のエンジンがある。該当構成は latency も精度も使えない"
touch logs/infer_trt10.done
exit 0
