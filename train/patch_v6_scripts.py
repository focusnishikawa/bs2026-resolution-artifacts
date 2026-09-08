#!/usr/bin/env python3
"""Phase 4b 用に v6 の学習・特徴抽出スクリプトへ解像度スイープと ViT-L 対応を足す.

元スクリプトは書き換えず、別名 (finetune_res_v6.py / extract_features_res_v6.py) で出力する。
v6 論文の出荷パイプラインを壊さないため。

  finetune_res_v6.py       : --downsample-to N / --native を追加
  extract_features_res_v6.py: 上記に加え --model に dinov2_l / dinov3_l を追加

条件 A = N へ落として 224 へ戻す (v6 既存の downsample 実験と同じ)
条件 B = N のまま入力する (Phase 2 の条件 B に対応)
"""
import os

R = "/home1/gfsi/ufsi0002/bs2026_v5_rebuild"

# ---------------------------------------------------------------- finetune
src = open(os.path.join(R, "scripts/finetune_resnet50_v5b.py")).read()

NEW_BT = '''def build_transforms(downsample_to=0, native=False):
    """v6 の前処理に解像度スイープを足したもの。

    downsample_to=0 なら従来どおり 224 で学習する。
    downsample_to=N のとき:
      native=False -> 224 クロップを N へ落としてから 224 へ戻す (条件 A: 劣化耐性)
      native=True  -> N のまま入力する (条件 B: 解像度別ネイティブ学習)
    """
    bicubic = transforms.InterpolationMode.BICUBIC
    norm = [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    tr = [transforms.Resize(256), transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    va = [transforms.Resize(256), transforms.CenterCrop(224)]
    if downsample_to > 0:
        s = int(downsample_to)
        deg = [transforms.Resize((s, s), interpolation=bicubic)]
        if not native:
            deg += [transforms.Resize((224, 224), interpolation=bicubic)]
        tr += deg
        va += deg
    return transforms.Compose(tr + norm), transforms.Compose(va + norm)
'''

old_bt = src[src.index("def build_transforms():"):src.index("def split_train_val(")]
src = src.replace(old_bt, NEW_BT + "\n\n")

# 読み込み失敗時のダミー画像を入力寸法に合わせる (224 固定だと native 時に壊れる)
src = src.replace("            return torch.zeros(3, 224, 224), 0, False",
                  "            return torch.zeros(3, self.input_res, self.input_res), 0, False")
src = src.replace("    def __init__(self, rows, transform, label_to_idx):",
                  "    def __init__(self, rows, transform, label_to_idx, input_res=224):\n"
                  "        self.input_res = input_res")

src = src.replace('    ap.add_argument("--seed", type=int, default=42)',
                  '    ap.add_argument("--seed", type=int, default=42)\n'
                  '    ap.add_argument("--downsample-to", type=int, default=0,\n'
                  '                    help="0=従来の 224 学習。N>0 で解像度スイープ")\n'
                  '    ap.add_argument("--native", action="store_true",\n'
                  '                    help="N のまま入力する (条件 B)。既定は 224 へ戻す (条件 A)")')

src = src.replace("    train_tf, val_tf = build_transforms()",
                  "    train_tf, val_tf = build_transforms(args.downsample_to, args.native)\n"
                  "    _ires = args.downsample_to if (args.native and args.downsample_to > 0) else 224\n"
                  "    print('[res] downsample_to=%d native=%s input_res=%d'\n"
                  "          % (args.downsample_to, args.native, _ires), flush=True)")

out1 = os.path.join(R, "scripts/finetune_res_v6.py")
open(out1, "w").write(src)
print("[ok] %s (%d bytes)" % (out1, len(src)))

# ---------------------------------------------------------------- extract
src2 = open(os.path.join(R, "scripts/extract_cnn_features_v2.py")).read()

# --model に ViT-L を追加
src2 = src2.replace('choices=["mobilenet_v3_small", "resnet50"])',
                    'choices=["mobilenet_v3_small", "resnet50", "dinov2_l", "dinov3_l"])')

# ViT-L の構築分岐を足す (timm。特徴は num_classes=0 で pooled 特徴 1024 次元)
anchor = '    elif name == "resnet50":'
vit_branch = '''    elif name in ("dinov2_l", "dinov3_l"):
        # Phase 4b: v6 のカスケードのバックボーンを自己教師あり ViT-L に差し替える。
        # 特徴次元は 2048 (ResNet50) ではなく 1024 になるので LightGBM は全段再学習が要る。
        import timm
        tname = ("vit_large_patch14_dinov2" if name == "dinov2_l"
                 else "vit_large_patch16_dinov3")
        ires = int(os.environ.get("V6_INPUT_RES", "224"))
        if name == "dinov2_l" and ires % 14 != 0:      # patch14 は 14 の倍数へ丸める
            ires = max(14, 14 * round(ires / 14))
        m = timm.create_model(tname, pretrained=False, num_classes=0, img_size=ires)
        return m, 1024
'''
src2 = src2.replace(anchor, vit_branch + anchor)

# native (N のまま入力) を足す
src2 = src2.replace('''        ops += [transforms.Resize((s, s), interpolation=bicubic),
                transforms.Resize((224, 224), interpolation=bicubic)]''',
                    '''        ops += [transforms.Resize((s, s), interpolation=bicubic)]
        if not getattr(args, "native", False):
            ops += [transforms.Resize((224, 224), interpolation=bicubic)]''')
src2 = src2.replace('    ap.add_argument("--downsample-to", type=int, default=0,',
                    '    ap.add_argument("--native", action="store_true",\n'
                    '                    help="N のまま入力する (条件 B)。既定は 224 へ戻す (条件 A)")\n'
                    '    ap.add_argument("--downsample-to", type=int, default=0,')

if "import os" not in src2.split("\n\n")[0]:
    src2 = src2.replace("import argparse", "import argparse\nimport os", 1)

out2 = os.path.join(R, "scripts/extract_features_res_v6.py")
open(out2, "w").write(src2)
print("[ok] %s (%d bytes)" % (out2, len(src2)))
