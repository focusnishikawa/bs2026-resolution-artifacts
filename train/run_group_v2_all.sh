#!/usr/bin/env bash
# 査読指摘 A-3 (v2 規則): 動画 ID を含む群定義での group split 再分割・再学習を
# 頭から尻まで 1 本で回すオーケストレータ (hpciaiss1 用).
#
#   Phase F  評価穴埋めパトロールを背景起動  fill_group_eval_v2.sh        (GPU 3・並走)
#   Phase M  本線 学習+評価 30 シード        run_group_split_boundary_v2.sh (GPU 0/1/2)
#   Phase W  パトロールの完了を待つ          logs/fill_group_eval_v2.done  (上限 6 時間)
#   Phase C  条件 A 用 dinov2_l_r224 追加学習 fill_group_condA_dinov2_v2.sh (1 巡)
#   Phase R  条件 A DINOv2-L 評価のリトライ  retry_group_condA_dinov2_v2.sh (最大 12 巡)
#
# ⭐ Phase F を**先に**背景で上げるのは v1 の運用と同じである。本線 (Phase M) は条件ごとに
#    3 プロセスを同時起動するせいで UCX の Caught signal 11 が出て評価が散発的に全滅する。
#    パトロールが GPU 3 で逐次に埋め直しながら並走する。
# ⭐ Phase C は **1 巡で 30/30 そろわないのが既知**なので、非ゼロで終わっても止めずに
#    Phase R (リトライ) へ進む。止めるのは Phase M と Phase R だけである。
# ⭐ 各サブスクリプトは冪等 (学習は test_metrics.json、評価は --skip_existing で判定) なので、
#    途中で落ちても同じコマンドを再投入すれば続きから進む。
#
# ⚠️ 前提: v2 規則の分割が先に要る。無ければ Phase F/M が即 abort する。
#      python3 train/make_group_split.py --rule v2 \
#          --out /work/gfsi/ufsi0002/bs2026-raptor/data/splits_group_v2
# ⚠️ v1 (A-3) の成果物とは出力先が完全に分かれている。上書きしない。
#      models_group_v2_s<seed>/ ・results/G2_cond{A,B}_s<seed>/ ・logs/*_v2*
# ⚠️ 本線の標準出力は tee で logs/group_boundary_v2_master.log にも落とす。
#    パトロール (fill_group_eval_v2.sh) が「本線が今どの seed を処理中か」を
#    そのログから読んでいるので、これを省くと本線と同じ seed を触りに行ってしまう。
#
# 起動:
#   cd /work/gfsi/ufsi0002/bs2026-resolution && nohup setsid bash train/run_group_v2_all.sh \
#       > logs/group_v2_all_master.log 2>&1 < /dev/null &
# 監視: tail -f logs/group_v2_all_master.log     完了マーカー: logs/group_v2_all.done

set -u

PROJ=/work/gfsi/ufsi0002/bs2026-resolution
SPG=/work/gfsi/ufsi0002/bs2026-raptor/data/splits_group_v2
S0=42; S1=71                  # 30 シード
N_BOUNDARY=16                 # 境界構成数 (ResNet50 11 + DINOv2-L 3 + ViT-S 2)
N_EXPECT=$(( (S1 - S0 + 1) * N_BOUNDARY ))   # 480

cd "$PROJ" || exit 1
mkdir -p logs results
rm -f "$PROJ/logs/group_v2_all.done"

say() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }

# 構成 JSON だけを数える (summary*.json を数えて 17/16 になった S15 の教訓)
count_json() {
    ls "$1" 2>/dev/null | grep -cE '^(resnet50|dinov2_l|vit_small)_r[0-9]+\.json$'
}

PH_NAME=(); PH_SEC=(); PH_RC=()

summary() {
    echo
    echo "----- フェーズ別 経過時間 -----"
    i=0
    while [ "$i" -lt "${#PH_NAME[@]}" ]; do
        printf "  %-34s rc=%-3s %5d 分\n" "${PH_NAME[$i]}" "${PH_RC[$i]}" "$(( PH_SEC[i] / 60 ))"
        i=$((i + 1))
    done
    echo "  合計 $(( ($(date +%s) - T_ALL0) / 60 )) 分 ($(( ($(date +%s) - T_ALL0) / 3600 )) 時間)"
}

# フェーズを 1 つ走らせる。第 1 引数=表示名、第 2 引数=期待する done マーカー ("" なら見ない)、
# 残り=実行するコマンド。非ゼロ終了か done マーカー欠落で 1 を返す。
run_phase() {
    local name="$1" marker="$2"; shift 2
    echo
    say "########## ${name} 開始  $(date '+%F %T') ##########"
    say "  cmd: $*"
    local t0 t1 rc
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    PH_NAME+=("$name"); PH_SEC+=("$((t1 - t0))"); PH_RC+=("$rc")
    say "########## ${name} 終了 rc=${rc}  $(( (t1 - t0) / 60 )) 分  $(date '+%F %T') ##########"
    if [ "$rc" -ne 0 ]; then
        return 1
    fi
    if [ -n "$marker" ] && [ ! -f "$marker" ]; then
        say "⚠️ ${name} は rc=0 だが完了マーカー ${marker} が無い"
        return 1
    fi
    return 0
}

# 止まったフェーズを明示して終了する。
die_at() {
    echo
    say "⛔⛔ $1 で停止した。後続フェーズは実行しない ⛔⛔"
    say "    ログ: logs/group_v2_all_master.log ・logs/group_boundary_v2_master.log"
    say "          logs/fill_group_eval_v2.log ・logs/fill_condA_dinov2_v2.log"
    summary
    exit 1
}

# ---- Phase M: 本線。tee で MASTER_LOG にも落とす (パトロールがそこを読む) ----
phase_main() {
    bash train/run_group_split_boundary_v2.sh $(seq "$S0" "$S1") 2>&1 \
        | tee "$PROJ/logs/group_boundary_v2_master.log"
    # ⚠️ パイプの後の $? は tee の結果。本線側の rc を見る
    return "${PIPESTATUS[0]}"
}

# ---- Phase W: パトロールの完了マーカーを 60 秒間隔で待つ (上限 6 時間) ----
#      パトロールが死んでいれば 6 時間待たずに落とす (48 時間上限や「埋まらない」exit 1 がある)
phase_wait_fill() {
    local w=0 max=360
    while [ ! -f "$PROJ/logs/fill_group_eval_v2.done" ]; do
        if ! pgrep -f 'fill_group_eval_v2\.sh' > /dev/null 2>&1; then
            # ⚠️ マーカーを touch した直後〜プロセス消滅までの隙間で見に行くと
            #    「死んだ」と誤判定する。もう一度マーカーを見てから落とす。
            sleep 5
            [ -f "$PROJ/logs/fill_group_eval_v2.done" ] && break
            say "  ⛔ パトロール (fill_group_eval_v2.sh) が完了マーカーを残さず終了している"
            say "     logs/fill_group_eval_v2.log の末尾を確認すること"
            return 1
        fi
        w=$((w + 1))
        [ "$w" -ge "$max" ] && { say "  ⛔ 6 時間待っても穴埋めが終わらない"; return 1; }
        [ $((w % 10)) -eq 1 ] && say "  [wait] 穴埋め待ち ${w} 分 ($(date '+%H:%M:%S'))"
        sleep 60
    done
    say "  穴埋め完了マーカーを確認した (${w} 分待った)"
    return 0
}

T_ALL0=$(date +%s)
say "##### group split v2 (動画 ID 込みの群定義) 一括実行 開始 #####"
say "  proj=${PROJ}  分割=${SPG}  seeds ${S0}-${S1}  境界 ${N_BOUNDARY} 構成"

# ---------- 前提チェック ----------
if [ ! -d "$SPG" ]; then
    say "[abort] v2 の分割が無い: $SPG"
    say "        先に: python3 train/make_group_split.py --rule v2 --out $SPG"
    exit 1
fi

# ---------- 競合チェック (自分自身は除く) ----------
# ⚠️ パターンに素の `run_group` を使うと **自スクリプト自身 (run_group_v2_all) に当たる**。
#    しかもコマンド置換で生まれる子シェルは親と同じ cmdline を持つので、PID で除くのも面倒になる。
#    そこで最初から自分に当たらない語で書き、保険として grep -v も掛ける。
#    (先行する本オーケストレータは、その配下の fill_group_eval_v2.sh が `fill_group_` で当たる)
BUSY=$(pgrep -af "run_phase2|eval_sweep|run_group_split_boundary|fill_group_|retry_group_condA" \
        2>/dev/null | grep -v "run_group_v2_all" || true)
if [ -n "$BUSY" ]; then
    say "[abort] 先行する学習・評価プロセスが動いている。GPU を取り合うので起動しない"
    echo "$BUSY" | sed 's/^/    /'
    exit 1
fi
say "  競合チェック: 先行プロセスなし"

# ---------- Phase F: 評価穴埋めパトロールを背景起動 ----------
echo
say "########## Phase F (穴埋めパトロール起動) ##########"
rm -f "$PROJ/logs/fill_group_eval_v2.done"
nohup setsid bash train/fill_group_eval_v2.sh > logs/fill_group_eval_v2.log 2>&1 < /dev/null &
FILL_PID=$!
sleep 5
say "  fill_group_eval_v2.sh を背景起動した (PID ${FILL_PID}・GPU 3・ログ logs/fill_group_eval_v2.log)"
say "  実プロセス: $(pgrep -f 'fill_group_eval_v2\.sh' | tr '\n' ' ')"
PH_NAME+=("Phase F (patrol 起動)"); PH_SEC+=("5"); PH_RC+=("0")

# ---------- Phase M: 本線 (学習 + 評価) ----------
run_phase "Phase M (本線 30 シード)" "logs/group_boundary_v2.done" phase_main \
    || die_at "Phase M (本線 30 シード)"

# ---------- Phase W: パトロールの完了待ち ----------
run_phase "Phase W (穴埋め完了待ち)" "logs/fill_group_eval_v2.done" phase_wait_fill \
    || die_at "Phase W (穴埋め完了待ち)"

# ---------- Phase C: 条件 A 用 dinov2_l_r224 の追加学習 + 1 巡目の評価 ----------
# ⚠️ 1 巡で 30/30 そろわないのは既知 (eval_sweep.py が約 50% の確率で rc=139)。
#    非ゼロで終わっても止めず、Phase R のリトライに任せる。
run_phase "Phase C (condA dinov2 1 巡)" "" bash train/fill_group_condA_dinov2_v2.sh \
    || say "  ⚠️ Phase C は非ゼロで終わった (1 巡で不足するのは既知)。Phase R へ進む"

# ---------- Phase R: 条件 A DINOv2-L 評価のリトライ ----------
run_phase "Phase R (condA retry)" "logs/fill_condA_dinov2_v2.done" \
          bash train/retry_group_condA_dinov2_v2.sh \
    || die_at "Phase R (condA retry)"

# ---------- 最終確認 ----------
echo
say "===== 成果物 (results/G2_cond{A,B}_s${S0}..${S1}) ====="
nA=0; nB=0; shortA=""; shortB=""
for s in $(seq "$S0" "$S1"); do
    a=$(count_json "$PROJ/results/G2_condA_s${s}")
    b=$(count_json "$PROJ/results/G2_condB_s${s}")
    nA=$((nA + a)); nB=$((nB + b))
    [ "$a" -ge "$N_BOUNDARY" ] || shortA="$shortA s${s}(${a})"
    [ "$b" -ge "$N_BOUNDARY" ] || shortB="$shortB s${s}(${b})"
done
say "  条件 A: ${nA}/${N_EXPECT}   条件 B: ${nB}/${N_EXPECT}   (${N_BOUNDARY} 構成 x $((S1 - S0 + 1)) シード)"
[ -n "$shortA" ] && say "  条件 A 未了:${shortA}"
[ -n "$shortB" ] && say "  条件 B 未了:${shortB}"
say "  学習モデル: $(ls -d "$PROJ"/models_group_v2_s* 2>/dev/null | wc -l | tr -d ' ') シード分のディレクトリ"

summary

if [ "$nA" -ge "$N_EXPECT" ] && [ "$nB" -ge "$N_EXPECT" ]; then
    echo
    say "##### group split v2 一括実行 完了 $(date '+%F %T') #####"
    say "  次: python3 train/collect_group_split.py --cond-prefix G2_cond \\"
    say "          --out $PROJ/results/summary_group_v2.json"
    touch "$PROJ/logs/group_v2_all.done"
    exit 0
fi
echo
say "⛔ そろわなかった (A ${nA}/${N_EXPECT} ・B ${nB}/${N_EXPECT})。完了マーカーは立てない"
say "   logs/fill_ev_*.log ・logs/group_boundary_v2_s*.log を確認すること"
exit 1
