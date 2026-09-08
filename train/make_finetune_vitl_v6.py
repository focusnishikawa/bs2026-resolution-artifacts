#!/usr/bin/env python3
"""v6 の fine-tune スクリプトから ViT-L 版 (finetune_vitl_v6.py) を生成する.

v6 の流儀は「head + layer4 のみ学習し、それ以外は凍結」なので、ViT-L でも
「head + 最終 n ブロック + final norm のみ学習」に対応させる (raptor の
train_dinov3_teacher.py の freeze_mode=partial と同じ考え方)。

差し替えるのは 2 か所だけ:
  build_model_finetune  -> timm の ViT-L を作り事前学習重みを読む
  optimizer の param 群 -> fc/layer4 ではなく head/最終ブロック

⚠️ DINOv2 は patch14 なので入力解像度を 14 の倍数へ丸める。
⚠️ DINOv2 は絶対位置埋め込みを持つので解像度に合わせて bicubic 補間する。
   DINOv3 は RoPE なので補間不要。
"""
import os

R = "/home1/gfsi/ufsi0002/bs2026_v5_rebuild"
src = open(os.path.join(R, "scripts/finetune_res_v6.py")).read()

VIT_BUILD = '''VITL_SPECS = {
    "dinov2_l": {"timm": "vit_large_patch14_dinov2", "patch": 14,
                 "weights": "/work/gfsi/ufsi0002/bs2026-raptor/weights/dinov2_vitl14_pretrain.pth"},
    "dinov3_l": {"timm": "vit_large_patch16_dinov3", "patch": 16,
                 "weights": "/work/gfsi/ufsi0002/bs2026-raptor/weights/dinov3_vitl16_lvd1689m.safetensors"},
}


def resolve_vit_input(model_id, res):
    """patch サイズで割り切れない解像度は倍数へ丸める (DINOv2 の patch14 用)。"""
    patch = VITL_SPECS[model_id]["patch"]
    if res % patch == 0:
        return res
    return max(patch, patch * round(res / patch))


def build_model_finetune(n_classes, model_id="resnet50", input_res=224, n_last_blocks=2):
    """v6 の流儀 (head + 最終層のみ学習) を ViT-L にも適用する。"""
    if model_id == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        for p in m.parameters():
            p.requires_grad = False
        for p in m.layer4.parameters():
            p.requires_grad = True
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        for p in m.fc.parameters():
            p.requires_grad = True
        return m

    import timm
    spec = VITL_SPECS[model_id]
    m = timm.create_model(spec["timm"], pretrained=False, num_classes=n_classes,
                          img_size=input_res)
    wp = spec["weights"]
    if os.path.exists(wp):
        if wp.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(wp)
        else:
            sd = torch.load(wp, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("head")}
        # DINOv2 は絶対位置埋め込みを持つので入力解像度に合わせて補間する
        if "pos_embed" in sd and getattr(m, "pos_embed", None) is not None \\
                and sd["pos_embed"].shape != m.pos_embed.shape:
            new_grid = int((m.pos_embed.shape[1] - 1) ** 0.5)
            from timm.layers import resample_abs_pos_embed
            sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"],
                                                     new_size=(new_grid, new_grid),
                                                     num_prefix_tokens=1)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        other = [k for k in missing if not k.startswith("head.")]
        print("[vitl] %s img_size=%d loaded (head_missing=%d other_missing=%d unexpected=%d)"
              % (model_id, input_res, len(missing) - len(other), len(other), len(unexpected)),
              flush=True)
        if other:
            raise RuntimeError("head 以外の重みが %d 件欠落: %s" % (len(other), other[:5]))
    else:
        raise FileNotFoundError(wp)

    # head + 最終 n ブロック + final norm のみ学習 (v6 の head+layer4 に対応)
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks[-n_last_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    for name, p in m.named_parameters():
        if name.startswith("head.") or name.startswith("norm.") or name.startswith("fc_norm."):
            p.requires_grad = True
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("[vitl] 学習対象 %.1fM / 全 %.1fM"
          % (n_tr / 1e6, sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    return m
'''

old = src[src.index("def build_model_finetune(n_classes):"):src.index("def all_reduce_mean(")]
src = src.replace(old, VIT_BUILD + "\n\n")

# optimizer: ResNet50 は fc/layer4、ViT-L は head/最終ブロック
src = src.replace('''    opt = torch.optim.AdamW([
        {"params": m_for_params.fc.parameters(), "lr": args.lr_head},
        {"params": m_for_params.layer4.parameters(), "lr": args.lr_layer4},
    ], weight_decay=args.weight_decay)''',
'''    if args.model_id == "resnet50":
        pg = [{"params": m_for_params.fc.parameters(), "lr": args.lr_head},
              {"params": m_for_params.layer4.parameters(), "lr": args.lr_layer4}]
    else:
        head_p = [p for n, p in m_for_params.named_parameters()
                  if p.requires_grad and (n.startswith("head.") or n.startswith("norm.")
                                          or n.startswith("fc_norm."))]
        blk_p = [p for n, p in m_for_params.named_parameters()
                 if p.requires_grad and n.startswith("blocks.")]
        pg = [{"params": head_p, "lr": args.lr_head},
              {"params": blk_p, "lr": args.lr_layer4}]
    opt = torch.optim.AdamW(pg, weight_decay=args.weight_decay)''')

# 引数追加
src = src.replace('    ap.add_argument("--native", action="store_true",',
                  '    ap.add_argument("--model-id", default="resnet50",\n'
                  '                    choices=["resnet50", "dinov2_l", "dinov3_l"])\n'
                  '    ap.add_argument("--n-last-blocks", type=int, default=2,\n'
                  '                    help="ViT-L で学習する最終ブロック数 (v6 の layer4 に相当)")\n'
                  '    ap.add_argument("--native", action="store_true",')

# build_model_finetune の呼び出しに引数を渡す
import re
src = re.sub(r"build_model_finetune\((\w+)\)",
             r"build_model_finetune(\1, args.model_id, _ires, args.n_last_blocks)", src)

# ViT-L のときは patch 倍数へ丸めた入力を使う
src = src.replace("    _ires = args.downsample_to if (args.native and args.downsample_to > 0) else 224",
                  "    _ires = args.downsample_to if (args.native and args.downsample_to > 0) else 224\n"
                  "    if args.model_id != 'resnet50':\n"
                  "        _ires = resolve_vit_input(args.model_id, _ires)")

out = os.path.join(R, "scripts/finetune_vitl_v6.py")
open(out, "w").write(src)
print("[ok] %s (%d bytes)" % (out, len(src)))
