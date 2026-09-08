#!/usr/bin/env python3
"""extract_features_res_v6.py の ViT-L 分岐を修正する.

⚠️ 元のパッチの誤り:
    elif name in ("dinov2_l", "dinov3_l"):
        m = timm.create_model(...)
        return m, 1024          <- ここで早期 return していた

  build_model の末尾には `feat_ext.to(device).eval()` があり、checkpoint のロードは
  各分岐の `_load_checkpoint(m, checkpoint)` が担う。早期 return するとその両方を
  飛ばしてしまい、**CPU 上の未学習モデル**に CUDA テンソルを渡すことになる。

  他の分岐と同じく feat_ext / dim を設定して末尾へ落ちる形に直す。
"""
import os

R = "/home1/gfsi/ufsi0002/bs2026_v5_rebuild"
p = os.path.join(R, "scripts/extract_features_res_v6.py")
src = open(p).read()

OLD = '''    elif name in ("dinov2_l", "dinov3_l"):
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

NEW = '''    elif name in ("dinov2_l", "dinov3_l"):
        # Phase 4b: v6 のカスケードのバックボーンを自己教師あり ViT-L に差し替える。
        # 特徴次元は 2048 (ResNet50) ではなく 1024 になるので LightGBM は全段再学習が要る。
        # ⚠️ ここで return してはいけない。末尾の feat_ext.to(device).eval() を通さないと
        #    CPU 上の未学習モデルに CUDA テンソルを渡すことになる。
        import timm
        tname = ("vit_large_patch14_dinov2" if name == "dinov2_l"
                 else "vit_large_patch16_dinov3")
        ires = int(os.environ.get("V6_INPUT_RES", "224"))
        if name == "dinov2_l" and ires % 14 != 0:      # patch14 は 14 の倍数へ丸める
            ires = max(14, 14 * round(ires / 14))
        # 学習時と同じ 895 クラスで作ってから重みを読み、その後 head を捨てて
        # pooled 特徴 (1024 次元) を出す形にする
        m = timm.create_model(tname, pretrained=False, num_classes=895, img_size=ires)
        if checkpoint:
            _load_checkpoint(m, checkpoint)
        m.reset_classifier(0)          # head を外して特徴抽出器にする
        print(f"  vit-l built: {tname} img_size={ires} -> feature 1024",
              file=sys.stderr)
        feat_ext = m
        dim = 1024
'''

assert OLD in src, "対象の分岐が見つからない (既に修正済み?)"
src = src.replace(OLD, NEW)
open(p, "w").write(src)
print("[ok] 修正: 早期 return を廃し、checkpoint ロードと device 転送を通すようにした")
