#!/usr/bin/env bash
# ViT-L の分割チェーンを TensorRT 10.3 で再ビルドする.
#
# 背景: JetPack 6.2 化で TRT が 8.5.2 -> 10.3.0 になり、既存エンジン 340 個が
#       magicTag 不一致で一切読めない。CNN 56 構成は build_trt10_cnn56.sh で再ビルド済みだが、
#       ViT-L の分割エンジンは 1 本も無い (engines_trt10/ に dino は 0 件)。
#       論文の Orin 値を全て新環境へ差し替える (ユーザー指示 2026-08-31) には ViT-L も要る。
#
# 配備構成はそのまま踏襲する (精度構成を変えると別物の測定になる):
#   DINOv2-L: 5 段。p0/p1/p2/p3s0 は FP16、p3s1 のみ FP32 強制
#             -> 最終ブロック群の活性が 3.6e5 に達し FP16 上限 65,504 を超えるため
#   DINOv3-L: 4 段すべて FP32 (段境界の残差ストリームが 1.55e5 で全段が超過)
#
# ⚠️ ONNX の置き場は N=32 とそれ以外で違う (配備構成の由来が違うため)。取り違えると
#    別のグラフを測ることになるので、下の onnx_v2 / onnx_v3 に集約してある。
#      DINOv2-L N=32 : onnx_split/dinov2_l_r32_p{0,1,2}.onnx
#                      onnx_split_p3/dinov2_l_r32_p3_static_p{0,1}.onnx
#      DINOv2-L N!=32: onnx_split_raw/dinov2_l_r<N>_fp16_p{0,1,2}.onnx
#                      onnx_split_raw_p3/dinov2_l_r<N>_fp16_p3_p{0,1}.onnx
#      DINOv3-L N=32 : onnx_split/dinov3_l_r32_p<k>_ort.onnx
#      DINOv3-L N!=32: onnx_split/dinov3_l_r<N>_fp16_sim_p<k>_ort.onnx
#
# ⚠️ TRT10 では --workspace が廃止された。build_trt10_cnn56.sh と同じく指定しない。
#
# usage:
#   bash build_trt10_vitl.sh check     # ONNX 在庫だけ確認 (GPU を使わない。測定中でも安全)
#   bash build_trt10_vitl.sh build     # 実ビルド
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
TRTEXEC=/usr/src/tensorrt/bin/trtexec
OUT=engines_trt10_split
RES="16 32 48 64 80 96 112 128 144 160 176 192 208 224"
CTRL_RES="16 32 64 112 128 224"          # 全段 FP32 対照を測る 6 水準 (図 4 の破線)
# ⛔ 2026-09-01: TRT 10.3 では `--precisionConstraints=obey --layerPrecisions=*:fp32` が
#    **警告ひとつ出さずに無視される**。旧 8.5.2 では効いていた (実測):
#      DINOv3-L r112 p0/p1/p2/p3 : 164.0/321.4/337.5/337.6 MB -> 85.0/161.0/169.7/169.1 MB (比 0.50-0.52)
#      DINOv2-L r112 p3s1        : 224.5 MB -> 73.1 MB (比 0.33)
#      同 p3s0 (FP32 指定なし)   : 72.6 MB -> 73.1 MB (比 1.01)  ← 対照。指定した段だけ半分になる
#    ログには LayerPrecisions: *:fp32 も "obey precision constraints" も出るのに適用されない。
#    結果 DINOv3-L は FP16 で数値が壊れ、全 14 解像度で出力が nan になった (相異なる出力 1/1882)。
#    ⭐ **全層 FP32 にする唯一確実な手段は `--fp16` を付けないこと。** PREC で切り替える。
FP32OPT=""                    # 旧: "--precisionConstraints=obey --layerPrecisions=*:fp32"
PREC_FP16="--fp16"
PREC_FP32=""                  # ⭐ --fp16 を付けない = FP32 でビルドされる
PREC="$PREC_FP16"             # build() が参照する既定精度

MODE="${1:-check}"
case "$MODE" in check|build) ;; *) echo "usage: bash $0 {check|build}"; exit 1 ;; esac

mkdir -p "$OUT" logs
[ "$MODE" = "build" ] && rm -f logs/build_trt10_vitl.done

# ---- 配備構成の ONNX 解決 ----
# onnx_v2 <res> <part>  -> DINOv2-L のパート ONNX パス
onnx_v2() {
    local r="$1" part="$2"
    if [ "$r" = "32" ]; then
        case "$part" in
            p3s0) echo "onnx_split_p3/dinov2_l_r32_p3_static_p0.onnx" ;;
            p3s1) echo "onnx_split_p3/dinov2_l_r32_p3_static_p1.onnx" ;;
            *)    echo "onnx_split/dinov2_l_r32_${part}.onnx" ;;
        esac
    else
        case "$part" in
            p3s0) echo "onnx_split_raw_p3/dinov2_l_r${r}_fp16_p3_p0.onnx" ;;
            p3s1) echo "onnx_split_raw_p3/dinov2_l_r${r}_fp16_p3_p1.onnx" ;;
            *)    echo "onnx_split_raw/dinov2_l_r${r}_fp16_${part}.onnx" ;;
        esac
    fi
}

# onnx_v3 <res> <k>  -> DINOv3-L のパート ONNX パス (全て ORT 最適化済みを使う)
onnx_v3() {
    local r="$1" k="$2"
    if [ "$r" = "32" ]; then
        echo "onnx_split/dinov3_l_r32_p${k}_ort.onnx"
    else
        echo "onnx_split/dinov3_l_r${r}_fp16_sim_p${k}_ort.onnx"
    fi
}

# onnx_v2_p3_whole <res> -> 全段 FP32 対照で使う「分割しない p3」
onnx_v2_p3_whole() {
    local r="$1"
    if [ "$r" = "32" ]; then echo "onnx_split/dinov2_l_r32_p3.onnx"
    else echo "onnx_split_raw/dinov2_l_r${r}_fp16_p3.onnx"; fi
}

# ---- ビルド 1 本 ----
# build <onnx> <engine> <log> [extra opts...]
built=0; skipped=0; failed=0; missing=0
build() {
    local onnx="$1" eng="$2" log="$3"; shift 3
    local name; name=$(basename "$eng" .engine)
    if [ ! -f "$onnx" ]; then
        echo "[$(date '+%H:%M:%S')] MISSING-ONNX ${name}  <- ${onnx}"
        missing=$((missing + 1)); return 1
    fi
    if [ -s "$eng" ]; then
        echo "[$(date '+%H:%M:%S')] skip         ${name} (既存)"
        skipped=$((skipped + 1)); return 0
    fi
    [ "$MODE" = "check" ] && { echo "[check] would build ${name}  <- ${onnx}"; return 0; }
    local t0 t1 rc sz
    t0=$(date +%s)
    # ⭐ $PREC は "--fp16" か "" (= FP32)。クォートしないので空なら引数ごと消える
    if timeout 3600 "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" $PREC "$@" > "$log" 2>&1; then
        t1=$(date +%s); sz=$(stat -c %s "$eng" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] OK           ${name}  $((t1-t0)) s  $((sz/1000000)) MB  [${PREC:-FP32}]"
        built=$((built + 1))
        # ⭐ FP32 のはずが FP16 に落ちていないかを毎回見る (これを見逃して 96 本を作り直した)。
        #    旧 8.5.2 のエンジンがあれば比べる。0.6 未満なら精度が落ちている疑いが濃い。
        if [ -z "$PREC" ]; then
            local osz ratio old
            for old in "engines_split/${name}.engine" "engines_split/${name}_fp32.engine"; do
                osz=$(stat -c %s "$old" 2>/dev/null) || continue
                [ -n "$osz" ] && [ "$osz" -gt 0 ] || continue
                ratio=$(awk -v a="$sz" -v b="$osz" 'BEGIN{printf "%.2f", a/b}')
                awk -v r="$ratio" 'BEGIN{exit !(r < 0.6)}' && \
                    echo "    ⚠️ FP32 指定なのに旧エンジンの ${ratio} 倍しかない (${old})。FP16 に落ちている疑い"
                break
            done
        fi
        return 0
    fi
    t1=$(date +%s)
    echo "[$(date '+%H:%M:%S')] FAILED       ${name}  $((t1-t0)) s -> ${log}"
    grep -aE "\[E\]|Killed|out of memory" "$log" | tail -2
    rm -f "$eng"; failed=$((failed + 1)); return 1
}

# ---- 競合チェック (測定・別ビルドと同時に走らせない) ----
if [ "$MODE" = "build" ]; then
    # ⚠️ pgrep は該当なしのとき「0」を出力しつつ終了コード 1 を返す。
    #    `$(pgrep -c ... || echo 0)` と書くと変数が "0\n0" の 2 行になり比較が壊れる
    #    (run_trt10_chain.sh の既知バグ)。必ず代入と既定値を分ける。
    n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then echo "[abort] trtexec が ${n} 本動作中"; exit 1; fi
fi

T0=$(date +%s)
echo "===== ViT-L 配備構成 126 パート (mode=${MODE}) ====="
for r in $RES; do
    echo "--- N=${r} ---"
    for part in p0 p1 p2 p3s0 p3s1; do
        # ⭐ p3s1 だけ FP32。build_p3s1_fp32.sh:16 に「FP16 では壊れる」と記録がある
        if [ "$part" = "p3s1" ]; then PREC="$PREC_FP32"; else PREC="$PREC_FP16"; fi
        build "$(onnx_v2 "$r" "$part")" "$OUT/dinov2_l_r${r}_${part}.engine" \
              "logs/btrt10_v2_r${r}_${part}.log"
    done
    # ⭐ DINOv3-L は全段 FP32。FP16 だと出力が nan になる (2026-09-01 に全 14 解像度で実測)
    PREC="$PREC_FP32"
    for k in 0 1 2 3; do
        build "$(onnx_v3 "$r" "$k")" "$OUT/dinov3_l_r${r}_p${k}.engine" \
              "logs/btrt10_v3_r${r}_p${k}.log"
    done
    PREC="$PREC_FP16"
done

echo
echo "===== 対照: DINOv2-L 全段 FP32 (6 水準 x 4 段) ====="
# 配備構成との違いは演算精度だけ。p3 を分割しないので 4 段になる
PREC="$PREC_FP32"   # ⭐ 対照そのものが「全段 FP32」なので FP16 に落ちてはいけない
for r in $CTRL_RES; do
    for part in p0 p1 p2; do
        build "$(onnx_v2 "$r" "$part")" "$OUT/dinov2_l_r${r}_f32_${part}.engine" \
              "logs/btrt10_v2f32_r${r}_${part}.log"
    done
    build "$(onnx_v2_p3_whole "$r")" "$OUT/dinov2_l_r${r}_f32_p3.engine" \
          "logs/btrt10_v2f32_r${r}_p3.log"
done
PREC="$PREC_FP16"

echo
echo "===== 参照: ViT-S/16 の FP32 エンジン (Orin FP32 対サーバの一致率用) ====="
PREC="$PREC_FP32"   # ⭐ FP16 との一致率を測る基準なので、これ自体が FP32 でなければ意味がない
for r in 112 224; do
    build "onnx/vit_small_r${r}.onnx" "$OUT/vit_small_r${r}_fp32.engine" \
          "logs/btrt10_vits_r${r}_fp32.log"
done
PREC="$PREC_FP16"

echo
echo "===== 検証: TRT 10.3 では分割せずに単体ビルドできるか (論文の貢献 4) ====="
# TRT 8.5.2 では Transformer 全体が 1 個の foreign node に融合され、その一括コンパイルが
# メモリ頂点を作るためビルドできなかった。10.3 で融合の粒度が変わっていれば
# 「分割しないと載らない」という前提そのものが変わる。必ず実測で確かめる。
for spec in "dinov2_l_r32:onnx/dinov2_l_r32.onnx" "dinov3_l_r32:onnx/dinov3_l_r32_ort.onnx"; do
    tag="${spec%%:*}"; onnx="${spec##*:}"
    if [ ! -f "$onnx" ]; then
        # 命名ゆれを吸収して探す (単体 ONNX は別名で置かれていることがある)
        alt=$(ls onnx/ 2>/dev/null | grep -E "^${tag}(_ort|_sim|_fp16_sim)?\.onnx$" | head -1)
        [ -n "$alt" ] && onnx="onnx/$alt"
    fi
    # ⭐⭐ 単体は両モデルとも FP32 で作る (2026-09-01 の実測で確定)。
    #    当初 DINOv2-L だけ FP16 で作っており、latency 8.049 ms = 「分割 8.657 ms より 7% 速い」
    #    という結果が出た。⛔ これは壊れたエンジンの値で、1,882 枚中 1,881 枚が同一 logits
    #    (acc 0.1201・サーバ一致 12.33%) だった。**計算が消えるぶん速く見えるだけ**である。
    #    FP32 で作り直すと 16.023 ms・サーバ一致 100.00%・相異なる出力 1,882/1,882 になり、
    #    エンジンは FP16 版のちょうど 1.99 倍 (1213.0 MB) になる。
    #    DINOv3-L は分割版でも FP16 が nan を返すので、もとより FP32。
    #    ⚠️ FP32 は重みが 2 倍になるので、ここで落ちること自体が「分割が要る」証拠になる
    PREC="$PREC_FP32"
    build "$onnx" "$OUT/${tag}_single.engine" "logs/btrt10_${tag}_single.log"
done
PREC="$PREC_FP16"

T1=$(date +%s)
echo
echo "===== 完了 $(date '+%F %T') mode=${MODE} ====="
echo "新規 ${built} / スキップ ${skipped} / 失敗 ${failed} / ONNX 欠 ${missing}"
echo "所要 $(( (T1-T0) / 60 )) 分"
echo "エンジン: $(ls "$OUT"/*.engine 2>/dev/null | wc -l) 本"
[ "$MODE" = "build" ] && touch logs/build_trt10_vitl.done
exit 0
