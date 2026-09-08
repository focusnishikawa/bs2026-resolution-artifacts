#!/usr/bin/env bash
# 査読指摘 A1 の再計算を 1 コマンドで通す.
#
#   fgpu0 から CSV 回収 -> 配備精度を集計 (test / val) -> 表とパレートを作り直す
#   -> 図 2 を描き直す -> **新旧の差分を出す**
#
# 前提: fgpu0 の測定 (run_30seed_chain_v2.sh) が終わっていること。
#       終わっていなくても走るが、そろっていないシードは警告付きの暫定値になる。
#
# 査読指摘との対応:
#   A1  報告する精度を **Orin の配備 FP16 エンジン**の実測へ差し替える
#   A5  構成選択は **validation**、報告は **test** (どちらも配備精度)
#   A-4 **ViT-L も配備エンジンで測る** (seed 43-71 の 29 本)。そろっていれば報告値に含め、
#       「ViT-L 級のみサーバ FP32」という例外を解消する
#
# usage:
#   bash train/refresh_a1.sh              # 回収から通す
#   SKIP_FETCH=1 bash train/refresh_a1.sh # 回収済みの CSV で集計だけやり直す
#   WITH_VITL=0 bash train/refresh_a1.sh  # ViT-L を含めない (従来の挙動)
#   WITH_VITL=1 bash train/refresh_a1.sh  # そろっていなくても ViT-L を含める (暫定確認用)
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R" || exit 1
SRC="${SRC:-trt10-maxn}"
FT="results/final_tables_${SRC//-/_}.json"
[ "$SRC" = "legacy" ] && FT="results/final_tables.json"
OLD="results/optimal_n_${SRC//-/_}.json"
[ "$SRC" = "legacy" ] && OLD="results/optimal_n.json"
NEW="results/optimal_n_a1.json"

echo "##### A1 再計算 $(date '+%F %T')  (データ源 $SRC) #####"

# ---- 0. 測定の完了確認 ----
# ⚠️ **マーカーだけを見てはいけない。** チェーン (run_30seed_chain_v2.sh) の
#    deploy_acc_30seed_v2_all.done は **seed 43-71 の 29 シード**で立つが、配備精度は
#    論文のサーバ FP32 と同じ **seed 42-71 の 30 シード**でそろえる。seed 42 は
#    run_seed42_chain.sh が第 2 段の後に測るので、マーカーが立った時点ではまだ 29 本しかない。
#    そこで**データ側 (そろったシードディレクトリ数) を数えて**判定する。
EXP_SEEDS=30
echo
echo "----- 0. fgpu0 の測定状態 (参考) -----"
# ⚠️⚠️ **ssh の失敗を「そろっていない」と同じ扱いにしてはいけない。**
#    2026-09-05 に実際に踏んだ: ControlMaster が一時的に切れて `Broken pipe` (rc=255) になり、
#    末尾の 2>/dev/null がそれを握り潰したため、**測定が全部終わっているのに test 0 / val 0** と
#    判定されて図が描かれなかった。再試行すれば 3 回とも 30 30 を返した。
#    → ①**3 回まで再試行**し ②**繋がらなかったことを明示**し
#      ③**図を描く判定はここではなく、回収済みのローカルデータで行う** (下記 §2c)
CNT=""
SSH_OK=0
for _try in 1 2 3; do
    CNT=$(ssh -o ConnectTimeout=25 ufsi0002@fgpu0 \
        'cd /home1/gfsi/ufsi0002/bs2026-resolution-edge 2>/dev/null || exit 1
         for p in preds_30seed preds_30seed_val; do
             n=0
             for d in "$p"/s*; do
                 [ -d "$d" ] && [ "$(ls "$d" 2>/dev/null | wc -l)" -ge 56 ] && n=$((n + 1))
             done
             printf "%s " "$n"
         done' 2>/dev/null)
    if [ -n "$CNT" ]; then SSH_OK=1; break; fi
    echo "   ⚠️ fgpu0 へ繋がらなかった (試行 $_try/3)"
    sleep 5
done
if [ "$SSH_OK" = "1" ]; then
    # ⚠️ 未クォート変数の単語分割に頼らない (zsh では分割されず 1 語になる)。awk で明示的に取る
    NT=$(printf '%s' "$CNT" | awk '{print $1+0}'); NT="${NT:-0}"
    NV=$(printf '%s' "$CNT" | awk '{print $2+0}'); NV="${NV:-0}"
    echo "   [fgpu0] 56 構成そろったシード: test $NT / $EXP_SEEDS   val $NV / $EXP_SEEDS"
    ssh -o ConnectTimeout=25 ufsi0002@fgpu0 \
        'cd /home1/gfsi/ufsi0002/bs2026-resolution-edge && \
         echo "   [チェーン] $(tail -1 logs/deploy_acc_30seed_v2_master.log)"; \
         echo "   [seed 42 ] $(tail -1 logs/seed42_master.log 2>/dev/null || echo 未起動)"' \
        || echo "   ⚠️ ログの取得には失敗した (判定には影響しない)"
else
    echo "   ⚠️⚠️ fgpu0 へ 3 回とも繋がらなかった。**回収済みのローカル CSV で判定する**"
fi

# ---- 1. 回収 ----
if [ "${SKIP_FETCH:-0}" != "1" ]; then
    echo
    echo "----- 1. CSV の回収 -----"
    bash train/fetch_30seed_preds.sh || exit 1
    # ⭐ ViT-L (査読指摘 A-4)。まだ 1 本も無くても失敗させない — A-4 の測定は
    #    CNN 側より後から始まっており、無い間は従来どおりサーバ FP32 で通す。
    bash train/fetch_vitl_preds.sh || echo "   ⚠️ ViT-L の回収に失敗した (A-4 は暫定のまま進む)"
else
    echo
    echo "----- 1. 回収はスキップ (SKIP_FETCH=1) -----"
fi

# ---- 1b. 回収済み CSV を数える関数 (図の可否判定と ViT-L の採否に使う) ----
# ⚠️ `local a="$1" b="$2"` と 1 行に並べない (S9 で set -u に弾かれた)
count_ready() {          # $1 = preds ディレクトリ / $2 = 1 シードあたりの構成数
    local root
    root="$1"
    local minn
    minn="$2"
    local n
    n=0
    local d
    for d in "$root"/s*; do
        [ -d "$d" ] || continue
        [ "$(ls "$d"/*.csv 2>/dev/null | wc -l)" -ge "$minn" ] && n=$((n + 1))
    done
    printf '%s' "$n"
}

# ---- 1c. ViT-L を報告値に含めるか (査読指摘 A-4) ----
# ⭐ ViT-L (dinov2_l / dinov3_l) の配備精度は別チェーン run_vitl_acc_seed.sh が
#    **seed 43-71 の 29 本**を測る。そろっていれば報告値に含め、「ViT-L 級のみサーバ FP32」
#    という例外を解消する。そろっていなければ optimal_n.py が従来どおりサーバ FP32 へ落とす。
# ⚠️ **ViT-L は 29 シード・CNN は 30 シードで母数が違う。**論文にはそう明記すること
#    (seed 42 の ViT-L は onnx_seeds/ でなく旧 onnx/ 由来で N=32 だけ別経路なので混ぜない)。
# ⚠️ val 側は ViT-L を測っていない。選択側は --select-fallback がサーバ FP32 の val へ
#    落とす既存の仕組みのままでよい (S13 で実装済み)。
EXP_VITL=29
LVL=$(count_ready results/preds_vitl_30seed 28)
VITL_ON=0
echo
echo "----- 1c. ViT-L の配備精度 (査読指摘 A-4) -----"
echo "   28 構成そろったシード: $LVL / $EXP_VITL"
case "${WITH_VITL:-auto}" in
    0)  echo "   → 含めない (WITH_VITL=0)" ;;
    1)  VITL_ON=1; echo "   → 含める (WITH_VITL=1 で強制)" ;;
    *)  if [ "$LVL" -ge "$EXP_VITL" ]; then
            VITL_ON=1; echo "   → 含める (29 シードそろった)"
        else
            echo "   → まだそろっていないので含めない。ViT-L はサーバ FP32 のまま"
            echo "      (強制するなら WITH_VITL=1。回収は bash train/fetch_vitl_preds.sh)"
        fi ;;
esac

# ---- 2. 配備精度の集計 ----
echo
echo "----- 2. 配備精度の集計 (test = 報告用) -----"
if [ "$VITL_ON" = "1" ]; then
    python3 train/collect_deploy_acc.py --preds results/preds_30seed \
        --split test \
        --models mnv4 effb0 resnet50 vit_small dinov2_l dinov3_l \
        --model-preds dinov2_l=results/preds_vitl_30seed \
                      dinov3_l=results/preds_vitl_30seed \
        --out results/summary_deploy.json
else
    python3 train/collect_deploy_acc.py --preds results/preds_30seed \
        --split test --out results/summary_deploy.json
fi
rc_t=$?

echo
echo "----- 2b. 配備精度の集計 (val = 構成選択用, 査読指摘 A5) -----"
python3 train/collect_deploy_acc.py --preds results/preds_30seed_val \
    --split val --out results/summary_deploy_val.json
rc_v=$?

if [ "$rc_t" -ne 0 ] || [ "$rc_v" -ne 0 ]; then
    echo
    echo "⚠️⚠️ シードがそろっていない構成がある (test rc=$rc_t / val rc=$rc_v)。"
    echo "     以下は**暫定値**である。測定完了後にもう一度流すこと"
fi

# ---- 2c. 図を描いてよいかの判定 (回収済みのローカルデータで行う) ----
# ⚠️ **判定を fgpu0 の応答に依存させない** (ssh 断で図が描かれない事故を 2026-09-05 に踏んだ)。
# ⚠️ **rc だけでは足りない。** collect_deploy_acc.py の既定 `need` は「見つかったシード数」なので、
#    11 シードしか無くても全構成が 11 そろっていれば rc=0 になる。**30 という絶対数を別に数える**。
LT=$(count_ready results/preds_30seed 56)
LV=$(count_ready results/preds_30seed_val 56)
echo
echo "----- 2c. 図の可否判定 (ローカル) -----"
echo "   56 構成そろったシード: test $LT / $EXP_SEEDS   val $LV / $EXP_SEEDS"
echo "   ViT-L 28 構成そろったシード: $LVL / $EXP_VITL  (報告値に含める = $VITL_ON)"
DONE_OK=0
if [ "$LT" -ge "$EXP_SEEDS" ] && [ "$LV" -ge "$EXP_SEEDS" ] \
   && [ "$rc_t" -eq 0 ] && [ "$rc_v" -eq 0 ]; then
    DONE_OK=1; echo "✅ 30 シード (seed 42-71) が test / val ともそろっている"
else
    echo "⚠️ まだそろっていない。以後は**暫定値**として扱う (図は描かない)"
fi
# ⚠️ ViT-L を含めるつもりで揃っていないときは、図を描く前に必ず止める。
#    CNN だけ 30 シードで ViT-L が 5 シード、という混ざった図を論文へ入れないため。
if [ "${WITH_VITL:-auto}" != "0" ] && [ "$LVL" -lt "$EXP_VITL" ] && [ "$DONE_OK" = "1" ]; then
    echo "⚠️⚠️ CNN 側は 30 シードそろっているが **ViT-L は $LVL / $EXP_VITL しか無い**。"
    echo "     このまま図を描くと ViT-L だけサーバ FP32 の図になる。A-4 の測定完了を待つこと"
    echo "     (待たずに CNN だけで描くなら WITH_VITL=0 FORCE_FIGS=1 を明示する)"
    DONE_OK=0
fi

# ---- 3. 表とパレートの再計算 ----
echo
echo "----- 3. 表・パレートの再計算 (A1 + A5) -----"
[ -f "$FT" ] || { echo "[中止] $FT が無い。先に analyze_all.py --src $SRC を実行すること"; exit 1; }
python3 train/optimal_n.py --src "$SRC" \
    --report-acc results/summary_deploy.json \
    --select-acc results/summary_deploy_val.json \
    --out "$NEW" || exit 1

# ---- 4. 図 2 の描き直し ----
echo
echo "----- 4. 図 2 (パレート) の描き直し -----"
# ⚠️⚠️ **測定が終わる前に描くと、暫定値で論文の図を上書きしてしまう。**
#      (実際に 2026-09-03 のテストで figs_en/fig2_pareto_orin.png を 11 シードの値で
#       上書きし、git checkout で戻す事故が起きた)
# ⚠️ 図 1 には効かせない (条件 A の配備精度は測っていないため。make_figs.py の注記を参照)
# ⚠️ PAPER_PDF=1 は付けない。論文は両版とも figs/*.png を参照しており PDF は使わない。
#    加えて **和文図の PDF 出力は matplotlib backend_pdf が日本語グリフ名を ascii へ
#    encode できず必ず落ちる** (既存不具合。HEAD でも同じ)。落ちると以降の図が生成されない。
if [ "$DONE_OK" = "1" ] || [ "${FORCE_FIGS:-0}" = "1" ]; then
    for lang in ja en; do
        if [ "$lang" = "en" ]; then export FIG_LANG=en; else unset FIG_LANG; fi
        ACC_JSON=results/summary_deploy.json FT_JSON="$FT" \
            python3 train/make_figs.py | grep -E "^\[A1\]|^ +⚠️|fig2" || true
    done
    unset FIG_LANG
else
    echo "⚠️ 測定が未完了なので**図は描かない** (暫定値で論文の図を上書きしないため)。"
    echo "   どうしても描くなら FORCE_FIGS=1 を付ける (描いた図は git checkout で戻せる)"
fi

# ---- 5. 新旧の差分 ----
echo
echo "----- 5. 新旧の差分 (論文のどこを書き換えるか) -----"
if [ -f "$OLD" ]; then
    python3 train/diff_a1.py --old "$OLD" --new "$NEW"
else
    echo "⚠️ 旧 $OLD が無いので差分は出せない"
fi

echo
echo "##### 完了 $(date '+%F %T') #####"
echo "  配備精度   : results/summary_deploy.json / results/summary_deploy_val.json"
echo "  表・パレート: $NEW"
echo "  差し替え箇所: docs/A1_replacement_map.md"
if [ "$VITL_ON" = "1" ]; then
    echo "  ViT-L      : **配備 FP16 の実測を含む** ($LVL シード。CNN の 30 シードとは母数が違う)"
else
    echo "  ViT-L      : サーバ FP32 のまま (A-4 の測定 $LVL/$EXP_VITL)"
fi
