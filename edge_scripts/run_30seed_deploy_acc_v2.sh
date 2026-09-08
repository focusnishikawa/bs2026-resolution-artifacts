#!/usr/bin/env bash
# 30 シードの「配備精度」を Orin 実機で測る — **test と validation の両方** (査読指摘 A1 + A5).
#
# v1 (run_30seed_deploy_acc.sh) との違いは 1 点だけである。
#
#   ⭐ **1 本のエンジンで test と val を続けて測る。**
#
#   v1 は test だけを測ってエンジンを即削除していた。val を後から別ジョブで測ると
#   **全エンジンを作り直すことになり、29 シードで約 46 時間が丸ごと増える**
#   (ビルド 1.8 分/構成に対し推論は 5 秒/構成。時間の 96% はビルドである)。
#   同じエンジンで続けて測れば、追加は推論分の約 1.6 時間で済む。
#
# なぜ val まで測るのか (ユーザー判断 2026-09-03):
#   査読指摘 A5 は「構成選択を test でやると選択バイアスが入る」であり、A1 は
#   「報告する精度は配備するエンジンのものでなければならない」である。両方を厳密に
#   満たすには **選択も報告も配備精度**で、かつ **選択は val・報告は test** である必要がある。
#   サーバ FP32 の val 精度で選ぶ代替案もあったが、FP16 で崩れるモデル (ViT-S/16) では
#   選択そのものが配備時に最適でなくなるため、val も実機で測る方を採った。
#
# 冪等性 (v1 から強化):
#   構成ごとに test CSV と val CSV の有無を別々に見る。
#     両方そろっている            -> ビルドしない (スキップ)
#     どちらか欠けている          -> ビルドして **欠けている方だけ**測る
#   したがって v1 が測り終えた test CSV は再利用され、val だけが追加で埋まる。
#   途中で落ちても同じコマンドで再開できる (書きかけ CSV は行数不一致で測り直される)。
#
# ⚠️ ViT-L は対象外 (v1 と同じ)。分割チェーンで 1 シード 126 パートになるため。
# ⚠️ seed 42 は preds_trt10/ で測定済みなので既定の対象は 43-71 の 29 シード。
#
# 起動: nohup setsid bash run_30seed_deploy_acc_v2.sh > logs/deploy_acc_30seed_v2_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/deploy_acc_30seed_v2_master.log
# 完了: logs/deploy_acc_30seed_v2.done
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
TRTEXEC=/usr/src/tensorrt/bin/trtexec
RES1=./orin_infer_res_trt10
# ⭐ 試験用に絞れる (例: MODELS=mnv4 RES="16 112" bash ... 43 43)
MODELS="${MODELS:-mnv4 effb0 resnet50 vit_small}"
RES="${RES:-16 32 48 64 80 96 112 128 144 160 176 192 208 224}"
S0="${1:-43}"; S1="${2:-71}"
ONNX_ROOT=onnx_seeds
TMP_ENG=engines_30seed_tmp

# 分割 (test / val) の定義。DO_SPLITS で片方だけに絞れる。
DO_SPLITS="${DO_SPLITS:-test val}"
IN_test=inputs;      OUT_test=preds_30seed;     N_test=1882
IN_val=inputs_val;   OUT_val=preds_30seed_val;  N_val=1876

mkdir -p "$TMP_ENG" logs
rm -f logs/deploy_acc_30seed_v2.done

[ -x "$RES1" ] || { echo "[abort] $RES1 が無い。先にコンパイルすること"; exit 1; }

# ⚠️ 競合チェックは測定を触る前に置く (逆だと走っている測定を壊す)
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then echo "[abort] trtexec が $n 本走っている。中止"; exit 1; fi

# ⚠️ 枚数はマニフェストを正とする。ここがずれると推論が途中で切れたり
#    ラベルとの対応がずれたりするので、思い込みの定数で走らせない。
for sp in $DO_SPLITS; do
    eval "d=\$IN_$sp"
    [ -d "$d" ] || { echo "[abort] 入力ディレクトリが無い: $d"; exit 1; }
    mf=$(ls "$d"/manifest_r*.json 2>/dev/null | head -1)
    if [ -n "$mf" ]; then
        nn=$(sed -n 's/.*"n"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "$mf" | head -1)
        if [ -n "$nn" ]; then
            eval "old=\$N_$sp"
            [ "$nn" != "$old" ] && echo "[info] $sp の枚数を $old -> $nn へ (出典 $mf)"
            eval "N_$sp=\$nn"
        fi
    fi
    eval "echo \"[入力] $sp: \$IN_$sp  枚数 \$N_$sp  -> \$OUT_$sp\""
    eval "mkdir -p \$OUT_$sp"
done

distinct() { tail -n +2 "$1" 2>/dev/null | cut -d, -f2- | sort -u | wc -l; }
# CSV が「ちゃんと出来ている」か。行数 = 枚数 + ヘッダ 1 行
have_csv() { [ -s "$1" ] && [ "$(wc -l < "$1")" -eq $(( $2 + 1 )) ]; }

echo "===== 30 シード配備精度 (test+val) seed ${S0}-${S1} 開始 $(date '+%F %T') ====="
echo "      分割: ${DO_SPLITS}"
"$TRTEXEC" --version 2>&1 | grep -i "TensorRT version" || true

tot_build=0; tot_skip_cfg=0; tot_ng=0; tot_const=0
tot_infer_t=0; tot_infer_v=0
T_ALL0=$(date +%s)

for s in $(seq "$S0" "$S1"); do
    od="$ONNX_ROOT/s$s"
    if [ ! -d "$od" ]; then
        echo "[$(date '+%H:%M:%S')] === s$s: ONNX が無い ($od) -> スキップ ==="
        tot_ng=$((tot_ng+56)); continue
    fi
    for sp in $DO_SPLITS; do eval "mkdir -p \$OUT_$sp/s$s"; done

    # この seed でまだ測っていない推論が何本あるか (test と val を別々に数える)
    n_need=0; have_t=0; have_v=0
    for m in $MODELS; do for r in $RES; do
        for sp in $DO_SPLITS; do
            eval "o=\$OUT_$sp/s$s/${m}_r${r}.csv"; eval "nn=\$N_$sp"
            if have_csv "$o" "$nn"; then
                [ "$sp" = "test" ] && have_t=$((have_t+1)) || have_v=$((have_v+1))
            else
                n_need=$((n_need+1))
            fi
        done
    done; done
    if [ "$n_need" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] === s$s: test $have_t / val $have_v そろい -> スキップ ==="
        tot_skip_cfg=$((tot_skip_cfg+56)); continue
    fi

    echo
    echo "[$(date '+%H:%M:%S')] ===== seed $s 開始 (既存 test $have_t / val $have_v・要測定 $n_need 本) ====="
    TS0=$(date +%s); s_build=0; s_ng=0; s_const=0; s_it=0; s_iv=0; s_skip=0

    for m in $MODELS; do
        for r in $RES; do
            # --- 何を測る必要があるか ---
            need=""
            for sp in $DO_SPLITS; do
                eval "o=\$OUT_$sp/s$s/${m}_r${r}.csv"; eval "nn=\$N_$sp"
                have_csv "$o" "$nn" || need="$need $sp"
            done
            if [ -z "$need" ]; then s_skip=$((s_skip+1)); continue; fi

            onnx="$od/${m}_r${r}.onnx"
            eng="$TMP_ENG/${m}_r${r}.engine"
            if [ ! -f "$onnx" ]; then
                echo "  [miss] s$s ${m}_r${r} (onnx が無い)"; s_ng=$((s_ng+1)); continue
            fi
            miss_bin=""
            for sp in $need; do
                eval "b=\$IN_$sp/inputs_r${r}.bin"
                [ -s "$b" ] || miss_bin="$miss_bin $sp"
            done
            if [ -n "$miss_bin" ]; then
                echo "  [miss] s$s ${m}_r${r} (入力 bin が無い:$miss_bin)"; s_ng=$((s_ng+1)); continue
            fi

            # --- ビルド (test と val で 1 本を共用する。ここが v2 の要点) ---
            t0=$(date +%s)
            if ! timeout 1800 "$TRTEXEC" --onnx="$onnx" --saveEngine="$eng" --fp16 \
                    > "logs/b30_s${s}_${m}_r${r}.log" 2>&1; then
                echo "  [FAIL build] s$s ${m}_r${r} $(( $(date +%s) - t0 ))s"
                rm -f "$eng"; s_ng=$((s_ng+1)); continue
            fi
            tb=$(( $(date +%s) - t0 )); s_build=$((s_build+1))

            # --- 必要な分割だけ推論する ---
            for sp in $need; do
                eval "b=\$IN_$sp/inputs_r${r}.bin"; eval "o=\$OUT_$sp/s$s/${m}_r${r}.csv"
                eval "nn=\$N_$sp"
                lg="logs/i30_s${s}_${m}_r${r}.log"
                [ "$sp" = "val" ] && lg="logs/i30v_s${s}_${m}_r${r}.log"
                "$RES1" "$eng" "$b" "$nn" "$r" "$o" > "$lg" 2>&1
                rc=$?
                d=$(distinct "$o")
                # ⭐ 壊れた FP16 エンジンは入力によらず定数を返し、計算が消えるぶん速く見える。
                #    latency では気づけないので、ここで相異なる出力を必ず数える。
                if [ "$rc" -ne 0 ]; then
                    echo "  [FAIL infer:$sp] s$s ${m}_r${r} rc=$rc"; s_ng=$((s_ng+1)); rm -f "$o"
                elif [ "$d" -le 1 ]; then
                    echo "  [CONST:$sp] s$s ${m}_r${r} 相異なる出力=${d}/${nn}  << 定数 (無効)"
                    s_const=$((s_const+1))
                    [ "$sp" = "test" ] && s_it=$((s_it+1)) || s_iv=$((s_iv+1))
                else
                    [ "$sp" = "test" ] && s_it=$((s_it+1)) || s_iv=$((s_iv+1))
                fi
            done
            # エンジンは 1 本ごとに消す (1 シード分ためると約 3 GB になる)
            rm -f "$eng"
            [ $(( (s_build + s_ng) % 14 )) -eq 0 ] && \
                printf "  [%s] s%-3s 進捗 build %2d/56 (test %2d / val %2d)  最後=%s_r%s build %ds\n" \
                       "$(date '+%H:%M:%S')" "$s" "$((s_build+s_ng))" "$s_it" "$s_iv" "$m" "$r" "$tb"
        done
    done

    tot_build=$((tot_build+s_build)); tot_ng=$((tot_ng+s_ng)); tot_const=$((tot_const+s_const))
    tot_infer_t=$((tot_infer_t+s_it)); tot_infer_v=$((tot_infer_v+s_iv))
    tot_skip_cfg=$((tot_skip_cfg+s_skip))
    printf "[%s] ===== seed %s 完了: build %d / 推論 test %d + val %d / NG %d / 定数 %d / %d 分 =====\n" \
           "$(date '+%H:%M:%S')" "$s" "$s_build" "$s_it" "$s_iv" "$s_ng" "$s_const" \
           "$(( ($(date +%s) - TS0) / 60 ))"
done

echo
echo "===== 全体完了 $(date '+%F %T') ====="
echo "ビルド ${tot_build} / 推論 test ${tot_infer_t} + val ${tot_infer_v} / 構成スキップ ${tot_skip_cfg} / 失敗 ${tot_ng} / 定数出力 ${tot_const}"
echo "所要 $(( ($(date +%s) - T_ALL0) / 3600 )) 時間"
for sp in $DO_SPLITS; do
    eval "o=\$OUT_$sp"
    echo "CSV ($sp): $(find "$o" -name '*.csv' | wc -l) 件"
done
[ "$tot_const" -gt 0 ] && echo "⛔ 定数出力がある。該当構成は精度として使えない"
rmdir "$TMP_ENG" 2>/dev/null
touch logs/deploy_acc_30seed_v2.done
exit 0
