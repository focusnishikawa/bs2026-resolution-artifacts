#!/usr/bin/env python3
"""finetune_vitl_v6.py / extract の patch14 丸め漏れを直す.

⚠️ 誤りの構造:
    317: train_tf, val_tf = build_transforms(args.downsample_to, args.native)   # 丸め前の値
    318: _ires = ...                                                            # ここで丸める
    320:     _ires = resolve_vit_input(args.model_id, _ires)
    349: model = build_model_finetune(..., _ires, ...)                          # 丸め後で構築

  transform は丸め前 (64/32)、モデルは丸め後 (70/28) で作られ、
  「Input height (64) doesn't match model (70)」で落ちる。

  DINOv3 は patch16 で 32/64/128/224 がすべて 16 の倍数なので丸めが起きず、
  5 本とも成功していたため気づけなかった。DINOv2 (patch14) だけが露見した。

  修正: 丸めを先に計算し、その値で transform を作る。
"""
import os

R = "/home1/gfsi/ufsi0002/bs2026_v5_rebuild"
p = os.path.join(R, "scripts/finetune_vitl_v6.py")
src = open(p).read()

OLD = """    train_tf, val_tf = build_transforms(args.downsample_to, args.native)
    _ires = args.downsample_to if (args.native and args.downsample_to > 0) else 224
    if args.model_id != 'resnet50':
        _ires = resolve_vit_input(args.model_id, _ires)"""

NEW = """    # ⚠️ 丸めを先に決めてから transform を作ること。
    # DINOv2 は patch14 なので 64 -> 70、32 -> 28 と入力寸法が変わる。
    # 順序を逆にすると transform が 64 のまま、モデルが 70 で作られて
    # 「Input height (64) doesn't match model (70)」で落ちる。
    _ires = args.downsample_to if (args.native and args.downsample_to > 0) else 224
    if args.model_id != 'resnet50':
        _ires = resolve_vit_input(args.model_id, _ires)
    # transform には丸め後の寸法を渡す (情報量は downsample_to、入力寸法は _ires)
    _tf_res = _ires if (args.native and args.downsample_to > 0) else args.downsample_to
    train_tf, val_tf = build_transforms(_tf_res, args.native)"""

assert OLD in src, "対象箇所が見つからない (既に修正済み?)"
src = src.replace(OLD, NEW)
open(p, "w").write(src)
print("[ok] finetune_vitl_v6.py: 丸めを transform より前に移動した")

# extract 側も同じ問題がないか確認する。こちらは V6_INPUT_RES で丸めた値を
# モデルに渡す一方、transform は --downsample-to をそのまま使う実装なので、
# DINOv2 では同じ食い違いが起きる。--downsample-to に丸め後の値を渡す運用にする。
print("[note] extract 側は run_v6_vitl.sh から丸め後の値を渡すよう呼び出しを直す")
