#!/usr/bin/env bash
# 30 シードの「配備精度」を Orin 実機で測る (査読指摘 A1 への対応).
#
# 背景:
#   論文の表 5 は **サーバ FP32 の 30 シード平均精度**と **Orin FP16 の速度**を
#   組み合わせていた。しかし seed 42 の実測で、ViT-S/16 は N>=96 で FP16 化により
#   最大 -10.4 ポイント劣化することが分かった (N=112: 0.8932 -> 0.8427)。
#   そこで **配備するエンジンそのものの精度**を 30 シードで測り直す。
#
# 手順 (seed ごとに):
#   ONNX (onnx_seeds/s<NN>) -> trtexec で FP16 エンジン -> 全数推論 1,882 枚 -> CSV
#   -> **エンジンは即削除** (1 シードあたり約 3 GB になるため)
#
# ⚠️ ViT-L は対象外。分割チェーンで 1 シード 126 パート、30 シードで 3,780 エンジンに
#    なるため。ViT-L は一致率 97.9-100% で半精度の影響が小さいことを seed 42 で実証済み。
#
# ⚠️ seed 42 は既に preds_trt10/ で測定済みなので既定の対象は 43-71 の 29 シード。
#
# 所要見込み: 1 シードあたりビルド約 1.5 時間 + 推論 5 分 -> 29 シードで約 46 時間。
# 起動: nohup setsid bash run_30seed_deploy_acc.sh > logs/deploy_acc_30seed_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/deploy_acc_30seed_master.log
# 完了: logs/deploy_acc_30seed.done
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
TRTEXEC=/usr/src/tensorrt/bin/trtexec
RES1=./orin_infer_res_trt10
NTEST=1882
# ⭐ 試験用に絞れるようにしてある (例: MODELS=mnv4 RES="16 112" bash ... 43 43)
MODELS="${MODELS:-mnv4 effb0 resnet50 vit_small}"
RES="${RES:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"
S0="${1:-43}"; S1="${2:-71}"
ONNX_ROOT=onnx_seeds
OUT_ROOT=preds_30seed
TMP_ENG=engines_30seed_tmp

mkdir -p "$OUT_ROOT" "$TMP_ENG" logs
rm -f logs/deploy_acc_30seed.done

[ -x "$RES1" ] || { echo "[abort] $RES1 が無い。先にコンパイルすること"; exit 1; }

# ⚠️ 競合チェックは電力モードや測定を触る前に置く (逆だと走っている測定を壊す)
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then echo "[abort] trtexec が $n 本走っている。中止"; exit 1; fi

distinct() { tail -n +2 "$1" 2>/dev/null | cut -d, -f2- | sort -u | wc -l; }

echo "===== 30 シード配備精度 seed ${S0}-${S1} 開始 $(date '+%F %T') ====="
"$TRTEXEC" --version 2>&1 | grep -i "TensorRT version" || true

tot_ok=0; tot_skip=0; tot_ng=0; tot_const=0
T_ALL0=$(date +%s)

for s in $(seq "$S0" "$S1"); do
    od="$ONNX_ROOT/s$s"
    cd_="$OUT_ROOT/s$s"
    mkdir -p "$cd_"

    n_want=$(( $(echo $MODELS | wc -w) * $(echo $RES | wc -w) ))
    n_csv=$(ls "$cd_"/*.csv 2>/dev/null | wc -l)
    if [ "$n_csv" -ge "$n_want" ]; then
        echo "[$(date '+%H:%M:%S')] === s$s: ${n_csv} CSV そろい -> スキップ ==="
        tot_skip=$((tot_skip+n_want)); continue
    fi
    if [ ! -d "$od" ]; then
        echo "[$(date '+%H:%M:%S')] === s$s: ONNX が無い ($od) -> スキップ ==="
        tot_ng=$((tot_ng+56)); continue
    fi

    echo
    echo "[$(date '+%H:%M:%S')] ===== seed $s 開始 (既存 CSV $n_csv/$n_want) ====="
    TS0=$(date +%s); s_ok=0; s_ng=0; s_const=0

    for m in $MODELS; do
        for r in $RES; do
            csv="$cd_/${m}_r${r}.csv"
            if [ -s "$csv" ] && [ "$(wc -l < "$csv")" -eq $((NTEST+1)) ]; then
                tot_skip=$((tot_skip+1)); continue
            fi
            onnx="$od/${m}_r${r}.onnx"
            bin="inputs/inputs_r${r}.bin"
            eng="$TMP_ENG/${m}_r${r}.engine"
            if [ ! -f "$onnx" ] || [ ! -s "$bin" ]; then
                echo "  [miss] s$s ${m}_r${r} (onnx か入力が無い)"; s_ng=$((s_ng+1)); continue
            fi

            t0=$(date +%s)
            if ! timeout 1800 "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 \
                    > "logs/b30_s${s}_${m}_r${r}.log" 2>&1; then
                echo "  [FAIL build] s$s ${m}_r${r} $(( $(date +%s) - t0 ))s"
                rm -f "$eng"; s_ng=$((s_ng+1)); continue
            fi
            tb=$(( $(date +%s) - t0 ))

            "$RES1" "$eng" "$bin" "$NTEST" "$r" "$csv" \
                > "logs/i30_s${s}_${m}_r${r}.log" 2>&1
            rc=$?
            d=$(distinct "$csv")
            # ⭐ 壊れた FP16 エンジンは入力によらず定数を返し、計算が消えるぶん速く見える。
            #    latency では気づけないので、ここで相異なる出力を必ず数える。
            if [ "$rc" -ne 0 ]; then
                echo "  [FAIL infer] s$s ${m}_r${r} rc=$rc"; s_ng=$((s_ng+1)); rm -f "$csv"
            elif [ "$d" -le 1 ]; then
                echo "  [CONST] s$s ${m}_r${r} 相異なる出力=${d}/${NTEST}  << 定数 (無効)"
                s_const=$((s_const+1)); s_ok=$((s_ok+1))
            else
                s_ok=$((s_ok+1))
            fi
            # エンジンは 1 本ごとに消す (1 シード分ためると約 3 GB になる)
            rm -f "$eng"
            [ $(( (s_ok + s_ng) % 14 )) -eq 0 ] && \
                printf "  [%s] s%-3s 進捗 %2d/56  最後=%s_r%s build %ds\n" \
                       "$(date '+%H:%M:%S')" "$s" "$((s_ok+s_ng))" "$m" "$r" "$tb"
        done
    done

    tot_ok=$((tot_ok+s_ok)); tot_ng=$((tot_ng+s_ng)); tot_const=$((tot_const+s_const))
    printf "[%s] ===== seed %s 完了: OK %d / NG %d / 定数 %d / %d 分 =====\n" \
           "$(date '+%H:%M:%S')" "$s" "$s_ok" "$s_ng" "$s_const" "$(( ($(date +%s) - TS0) / 60 ))"
done

echo
echo "===== 全体完了 $(date '+%F %T') ====="
echo "成功 ${tot_ok} / スキップ ${tot_skip} / 失敗 ${tot_ng} / 定数出力 ${tot_const}"
echo "所要 $(( ($(date +%s) - T_ALL0) / 3600 )) 時間"
echo "CSV: $(find "$OUT_ROOT" -name '*.csv' | wc -l) 件"
[ "$tot_const" -gt 0 ] && echo "⛔ 定数出力がある。該当構成は精度として使えない"
rmdir "$TMP_ENG" 2>/dev/null
touch logs/deploy_acc_30seed.done
exit 0
