#!/usr/bin/env bash
# 第 3 段: FP32 指定が効かず FP16 で作られた 97 本を作り直し、推論と測定をやり直す.
#
# ⛔ 背景 (2026-09-01 に実測で判明)
#   TRT 10.3 では `--precisionConstraints=obey --layerPrecisions=*:fp32` が
#   **警告ひとつ出さずに無視される**。旧 8.5.2 では効いていた。証拠:
#     DINOv3-L r112 p0-p3 : 164.0/321.4/337.5/337.6 MB -> 85.0/161.0/169.7/169.1 MB (比 0.50-0.52)
#     DINOv2-L r112 p3s1  : 224.5 -> 73.1 MB (比 0.33)
#     同     r112 p3s0    : 72.6 -> 73.1 MB (比 1.01)  ← FP32 指定なし = 変化なし (対照)
#   結果 DINOv3-L は FP16 で数値が壊れ、全 14 解像度で出力が nan になった。
#   全層 FP32 にする唯一確実な手段は **--fp16 を付けないこと** (build_trt10_vitl.sh の PREC)。
#
# 作り直す 97 本 = DINOv3-L 56 + DINOv2-L p3s1 14 + 全段FP32対照 24 + ViT-S/16 FP32 2 + 単体 1
# 作り直さない 57 本 = DINOv2-L p0/p1/p2/p3s0 56 + DINOv2-L 単体 1 (元から FP16 で正しい)
#
# 監視:   tail -f logs/trt10_stage3_master.log
# 完了:   logs/trt10_stage3.done
# 再開:   同じコマンドで再投入すれば済んだ段は飛ばす
set -u

E=/home1/gfsi/ufsi0002/bs2026-resolution-edge
cd "$E" || exit 1
BROKEN=engines_trt10_split_fp16_broken
LIST=fp32_engines.txt

# ⚠️ 第 2 段では DINOv3-L を測定から外していた。第 3 段では作り直して測るので必ず空にする
SKIP_MODELS=""; export SKIP_MODELS

say() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }
mkdir -p logs
rm -f logs/trt10_stage3.done

say "===== 第 3 段チェーン開始 ====="

# ---------- 0. 段 6 (MAXN deploy) の完了を待つ ----------
if [ ! -f logs/meas_trt10_vitl_maxn_deploy.done ]; then
    say "段 6 (MAXN deploy) の完了を待機"
    w=0
    while [ ! -f logs/meas_trt10_vitl_maxn_deploy.done ]; do
        sleep 60; w=$((w + 1))
        if [ $((w % 30)) -eq 0 ]; then
            say "  待機 ${w} 分 / maxn=$(ls results_trt10_vitl_maxn/*.json 2>/dev/null | wc -l)/70"
        fi
        if [ "$w" -gt 480 ]; then say "[abort] 段 6 が 8 時間経っても終わらない"; exit 1; fi
    done
    say "段 6 の完了を確認"
fi

# ⚠️ pgrep は該当なしのとき「0」を出力しつつ終了コード 1 を返す。代入と既定値を分ける
n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
if [ "$n" -gt 0 ]; then
    say "trtexec が ${n} 本残っている。10 分待つ"
    sleep 600
    n=$(pgrep -c trtexec 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then say "[abort] trtexec が ${n} 本動作中"; exit 1; fi
fi

# ---------- 1. FP16 で作られた 97 本を退避 ----------
if [ ! -f logs/stage3_quarantine.done ]; then
    say "----- 1. FP32 指定が効かなかった 97 本を退避 -----"
    if [ ! -s "$LIST" ]; then say "[abort] ${LIST} が無い"; exit 1; fi
    mkdir -p "$BROKEN"
    mv_n=0
    while read -r e; do
        [ -n "$e" ] || continue
        if [ -s "engines_trt10_split/${e}.engine" ]; then
            mv "engines_trt10_split/${e}.engine" "$BROKEN/${e}.engine" && mv_n=$((mv_n + 1))
        fi
    done < "$LIST"
    say "  退避 ${mv_n} 本 -> ${BROKEN}/"
    say "  残り $(ls engines_trt10_split/*.engine 2>/dev/null | wc -l) 本 (FP16 のままで正しい分)"
    touch logs/stage3_quarantine.done
fi

# ---------- 2. 再ビルド ----------
if [ ! -f logs/stage3_build.done ]; then
    say "----- 2. 97 本を FP32 で再ビルド (見込み 2-3 時間) -----"
    rm -f logs/build_trt10_vitl.done
    bash build_trt10_vitl.sh build > logs/build_trt10_vitl_fp32.log 2>&1
    say "  エンジン $(ls engines_trt10_split/*.engine 2>/dev/null | wc -l)/154 本"
    touch logs/stage3_build.done
fi

# ---------- 3. FP32 化の検証 (ここで止めるのが最重要) ----------
say "----- 3. FP32 化の検証 (退避した FP16 版とのサイズ比) -----"
bad=0; okn=0; miss=0
while read -r e; do
    [ -n "$e" ] || continue
    o=$(stat -c %s "$BROKEN/${e}.engine" 2>/dev/null || echo 0)
    m=$(stat -c %s "engines_trt10_split/${e}.engine" 2>/dev/null || echo 0)
    if [ "$m" -eq 0 ]; then echo "  MISSING  ${e}"; miss=$((miss + 1)); continue; fi
    if [ "$o" -eq 0 ]; then okn=$((okn + 1)); continue; fi
    r=$(awk -v a="$m" -v b="$o" 'BEGIN{printf "%.2f", a/b}')
    if awk -v r="$r" 'BEGIN{exit !(r < 1.5)}'; then
        echo "  ⚠️ ${e}: FP16 版の ${r} 倍しかない = FP32 化に失敗"
        bad=$((bad + 1))
    else
        okn=$((okn + 1))
    fi
done < "$LIST"
say "  FP32 化 OK ${okn} / 失敗 ${bad} / 欠 ${miss}"
if [ "$bad" -gt 0 ] || [ "$miss" -gt 0 ]; then
    say "[abort] FP32 化に失敗した構成がある。測定へ進まない"
    exit 1
fi

# ---------- 4. 全数推論をやり直す ----------
if [ ! -f logs/stage3_infer.done ]; then
    say "----- 4. 全数推論をやり直す -----"
    mkdir -p preds_trt10_fp16_broken
    # 作り直したエンジンを通る CSV を退避する。DINOv2-L は p3s1 を含むのでチェーン全体が対象
    for f in preds_trt10/dinov3_l_*.csv preds_trt10/dinov2_l_*.csv preds_trt10/vit_small_r*_FP32.csv; do
        [ -e "$f" ] && mv "$f" preds_trt10_fp16_broken/
    done
    say "  退避 $(ls preds_trt10_fp16_broken/*.csv 2>/dev/null | wc -l) 件"
    rm -f logs/infer_trt10.done
    bash infer_trt10_all.sh all > logs/infer_trt10_fp32.log 2>&1
    touch logs/stage3_infer.done
fi

# ---------- 5. nan / 定数出力が消えたか ----------
say "----- 5. 定数出力の確認 -----"
const=0; checked=0
for f in preds_trt10/dinov3_l_*.csv preds_trt10/dinov2_l_*.csv; do
    [ -e "$f" ] || continue
    checked=$((checked + 1))
    d=$(tail -n +2 "$f" | sort -u | wc -l)
    if [ "$d" -le 1 ]; then
        echo "  ⚠️ $(basename "$f"): 相異なる出力 ${d}"
        const=$((const + 1))
    fi
done
say "  検査 ${checked} 件 / 定数出力 ${const} 件"
if [ "$const" -gt 0 ]; then
    say "[abort] まだ定数出力がある。測定しても無意味なので止める"
    exit 1
fi

# ---------- 6. 作り直した構成の測定値を退避 ----------
if [ ! -f logs/stage3_purge.done ]; then
    say "----- 6. 作り直した構成の測定値を退避 -----"
    mkdir -p results_trt10_vitl_fp16_broken
    for mode in 15w maxn; do
        for f in "results_trt10_vitl_${mode}"/dinov2_l_r*_p3s1.json "results_trt10_vitl_${mode}"/dinov3_l_*.json; do
            [ -e "$f" ] && mv "$f" "results_trt10_vitl_fp16_broken/${mode}_$(basename "$f")"
        done
    done
    say "  退避 $(ls results_trt10_vitl_fp16_broken/*.json 2>/dev/null | wc -l) 件"
    # 測定側の完了マーカーも消す (残っていると別経路で「済み」と誤認されうる)
    rm -f logs/meas_trt10_vitl_15w_deploy.done logs/meas_trt10_vitl_maxn_deploy.done \
          logs/meas_trt10_vitl_maxn_ctrl.done
    touch logs/stage3_purge.done
fi

# ---------- 7-9. 測定 ----------
run_meas() {
    local m="$1" w="$2"
    if [ -f "logs/stage3_meas_${m}_${w}.done" ]; then
        say "  ${m} ${w} は完了済み"
        return 0
    fi
    say "----- 測定 ${m} ${w} (欠けている分だけ測る) -----"
    bash measure_trt10_vitl_mode.sh "$m" "$w" > "logs/meas_trt10_vitl_${m}_${w}_fp32.log" 2>&1
    say "  ${m}: $(ls "results_trt10_vitl_${m}"/*.json 2>/dev/null | wc -l) 件"
    touch "logs/stage3_meas_${m}_${w}.done"
}
run_meas 15w deploy
run_meas maxn deploy
run_meas maxn ctrl

# ---------- 10. アイドル電力 (tegratop を止めた状態で測り直す) ----------
if [ ! -f logs/stage3_idle.done ]; then
    say "----- 10. アイドル電力を測り直す -----"
    # ⚠️ 前回は tegratop が動いた状態で測った。止まっていることは呼び出す側で確認済み
    bash idle_power.sh 60 > logs/idle_power_fp32.log 2>&1
    tail -3 logs/idle_power_fp32.log
    touch logs/stage3_idle.done
fi

# ---------- 11. 前処理を両モードで測り直す ----------
if [ ! -f logs/stage3_prep.done ]; then
    say "----- 11. 前処理を両電力モードで測り直す -----"
    bash measure_prep_mode.sh all > logs/prep_mode_master.log 2>&1
    say "  JSON $(ls results_prep_trt10/*.json 2>/dev/null | wc -l)/6 件"
    touch logs/stage3_prep.done
fi

# ---------- 終了処理 ----------
say "----- 電力モードを主モード (MAXN_SUPER) に確定 -----"
sudo -n nvpmodel -m 2 2>&1 | head -2
sleep 5
nvpmodel -q 2>&1 | head -2

say "===== 第 3 段チェーン完了 ====="
say "  エンジン $(ls engines_trt10_split/*.engine 2>/dev/null | wc -l)/154"
say "  15w $(ls results_trt10_vitl_15w/*.json 2>/dev/null | wc -l)/126"
say "  maxn $(ls results_trt10_vitl_maxn/*.json 2>/dev/null | wc -l)/152"
touch logs/trt10_stage3.done
exit 0
