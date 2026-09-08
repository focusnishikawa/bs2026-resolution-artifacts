#!/usr/bin/env python3
"""bs2026-resolution: 入力解像度 N x N を軸にした種判別モデルの学習.

設計は docs/DESIGN.md を参照。要点のみ:

  元画像 --[Resize(256) -> CenterCrop(224)]--> 224 基準クロップ
         --[BICUBIC 縮小]--> N x N            ... これが「解像度 N」の定義 (情報量)
         --[mode A のみ: BICUBIC で 224 へ再拡大]--> モデル入力

  mode B (native)  : N x N をそのままモデルへ入力し、その解像度で学習する (本スクリプトの主用途)
  mode A (restore) : 224 へ復元して入力する。学習は N=224 と等価なので通常は使わないが、
                     条件 A の学習側統制が必要になった場合のために用意する

DINOv2-L のみ patch 14 のため、ネイティブ入力解像度は 14 の倍数へ丸める (N' = 14 * round(N/14))。
丸めの実際値は meta.json / test_metrics.json の input_res に必ず記録する。

usage (smoke):
    python3 train/train_res.py --model effb0 --res 32 --epochs 1 --dry_run \\
        --data_root <...> --splits_dir <...> --output <...>

usage (full):
    python3 train/train_res.py --model dinov3_l --res 112 \\
        --data_root <...> --splits_dir <...> --pretrain_dir <...> --output <...>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

_nullcontext = contextlib.nullcontext

BASE_RES = 224  # 情報量の基準となる中央クロップ寸法
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================== モデル定義 ==============================

# kind:
#   cnn_timm  = timm + HF キャッシュの pretrained
#   cnn_tv    = torchvision + torch hub キャッシュの pretrained (v6 と揃えるため ResNet50 のみ)
#   vit_timm  = timm + HF キャッシュ。img_size を渡すと pos_embed は timm が自動 resample
#   vit_l     = timm skeleton (pretrained=False) + 重みファイルを手動ロード
MODEL_SPECS = {
    "mnv4": {
        "timm_name": "mobilenetv4_conv_medium.e500_r256_in1k",
        "kind": "cnn_timm", "patch": None,
        "lr_backbone": 5e-4, "lr_head": 5e-3, "epochs": 80, "batch_size": 256,
        "warmup_epochs": 5, "patience": 10, "amp": True, "llrd": None,
    },
    "effb0": {
        "timm_name": "efficientnet_b0.ra_in1k",
        "kind": "cnn_timm", "patch": None,
        "lr_backbone": 5e-4, "lr_head": 5e-3, "epochs": 80, "batch_size": 256,
        "warmup_epochs": 5, "patience": 10, "amp": True, "llrd": None,
    },
    "resnet50": {
        "timm_name": "torchvision:resnet50:IMAGENET1K_V2",
        "kind": "cnn_tv", "patch": None,
        "lr_backbone": 5e-4, "lr_head": 5e-3, "epochs": 80, "batch_size": 256,
        "warmup_epochs": 5, "patience": 10, "amp": True, "llrd": None,
    },
    "vit_small": {
        "timm_name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "kind": "vit_timm", "patch": 16,
        "lr_backbone": 1e-4, "lr_head": 1e-3, "epochs": 80, "batch_size": 256,
        "warmup_epochs": 10, "patience": 15, "amp": True, "llrd": None,
    },
    "dinov2_l": {
        "timm_name": "vit_large_patch14_dinov2",
        "kind": "vit_l", "patch": 14, "pretrain_file": "dinov2_vitl14_pretrain.pth",
        "lr_backbone": 5e-5, "lr_head": 5e-4, "epochs": 30, "batch_size": 64,
        "warmup_epochs": 3, "patience": 5, "amp": False, "llrd": 0.75,
    },
    "dinov3_l": {
        "timm_name": "vit_large_patch16_dinov3",
        "kind": "vit_l", "patch": 16, "pretrain_file": "dinov3_vitl16_lvd1689m.safetensors",
        "lr_backbone": 5e-5, "lr_head": 5e-4, "epochs": 30, "batch_size": 64,
        "warmup_epochs": 3, "patience": 5, "amp": False, "llrd": 0.75,
    },
}

RESOLUTIONS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]


def resolve_input_res(model_id: str, res: int, mode: str) -> int:
    """モデルへ実際に入力するテンソル寸法を返す。

    mode A は常に BASE_RES。mode B は res をそのまま使うが、patch サイズで割り切れない
    モデル (DINOv2-L の patch 14) は最も近い patch 倍数へ丸める (下限は 1 patch)。
    """
    if mode == "A":
        return BASE_RES
    patch = MODEL_SPECS[model_id]["patch"]
    if patch is None or res % patch == 0:
        return res
    return max(patch, patch * round(res / patch))


# ============================== 前処理 ==============================

class DegradeResolution:
    """BASE_RES の PIL 画像を res へ BICUBIC 縮小し、restore_to があればそこへ再拡大する。

    「解像度 N」の定義そのもの。条件 A と条件 B で情報の落ち方を完全に一致させるため、
    どちらも必ずこのクラスを通す。
    """

    def __init__(self, res: int, restore_to: int | None = None):
        self.res = res
        self.restore_to = restore_to

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), Image.BICUBIC)
        if self.restore_to is not None and self.restore_to != self.res:
            img = img.resize((self.restore_to, self.restore_to), Image.BICUBIC)
        return img

    def __repr__(self):
        return f"DegradeResolution(res={self.res}, restore_to={self.restore_to})"


def build_transforms(res: int, input_res: int):
    """aug は常に BASE_RES 上で行い、その後に解像度を落とす。

    こうすると解像度以外の条件 (aug の強度、クロップ範囲) が全水準で不変になる。
    restore_to は「落とした後に入力寸法へ戻す」ためのもので、
    mode A では 224、mode B で patch 丸めが起きた場合は丸め後の寸法になる。
    """
    restore_to = input_res if input_res != res else None
    degrade = DegradeResolution(res, restore_to=restore_to)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(BASE_RES, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        degrade,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(BASE_RES * 1.143)),   # 256
        transforms.CenterCrop(BASE_RES),
        degrade,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_tf, val_tf


class SpeciesDataset(Dataset):
    """rel_path / species_idx を持つ split CSV を読む単純な分類用 Dataset。"""

    def __init__(self, csv_path: Path, data_root: Path, transform):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_root / row["rel_path"]).convert("RGB")
        return self.transform(img), int(row["species_idx"])


# ============================== モデル構築 ==============================

def build_model(model_id: str, num_classes: int, input_res: int, pretrain_dir: Path | None,
                pretrained: bool = True):
    """(model, info) を返す。info は重みロード状況の記録用。"""
    spec = MODEL_SPECS[model_id]
    kind = spec["kind"]
    info = {"timm_name": spec["timm_name"], "kind": kind, "input_res": input_res}

    if kind == "cnn_timm":
        import timm
        model = timm.create_model(spec["timm_name"], pretrained=pretrained, num_classes=num_classes)

    elif kind == "cnn_tv":
        import torchvision.models as tvm
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        info["weights"] = "IMAGENET1K_V2" if pretrained else None

    elif kind == "vit_timm":
        import timm
        # img_size を渡すと timm が pretrained の pos_embed を自動 resample する
        model = timm.create_model(spec["timm_name"], pretrained=pretrained,
                                  num_classes=num_classes, img_size=input_res)

    elif kind == "vit_l":
        import timm
        model = timm.create_model(spec["timm_name"], pretrained=False,
                                  num_classes=num_classes, img_size=input_res)
        if pretrained:
            if pretrain_dir is None:
                raise ValueError(f"{model_id} には --pretrain_dir が必要")
            path = Path(pretrain_dir) / spec["pretrain_file"]
            if not path.exists():
                raise FileNotFoundError(f"pretrain not found: {path}")
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file
                sd = load_file(str(path))
            else:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = {k: v for k, v in sd.items() if not k.startswith("head")}

            # DINOv2 は絶対位置埋め込みを持つため入力解像度に合わせて補間する。
            # DINOv3 は RoPE なので pos_embed 自体が存在せず、この分岐に入らない。
            if "pos_embed" in sd and hasattr(model, "pos_embed") and model.pos_embed is not None \
                    and sd["pos_embed"].shape != model.pos_embed.shape:
                new_grid = int((model.pos_embed.shape[1] - 1) ** 0.5)
                try:
                    from timm.layers import resample_abs_pos_embed
                    sd["pos_embed"] = resample_abs_pos_embed(
                        sd["pos_embed"], new_size=(new_grid, new_grid), num_prefix_tokens=1)
                except ImportError:
                    old = sd["pos_embed"]
                    cls_tok, patch_tok = old[:, :1], old[:, 1:]
                    old_grid = int(patch_tok.shape[1] ** 0.5)
                    D = patch_tok.shape[-1]
                    patch_tok = patch_tok.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
                    patch_tok = F.interpolate(patch_tok, size=(new_grid, new_grid),
                                              mode="bicubic", align_corners=False)
                    patch_tok = patch_tok.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
                    sd["pos_embed"] = torch.cat([cls_tok, patch_tok], dim=1)
                info["pos_embed_resampled_to"] = int(new_grid)

            missing, unexpected = model.load_state_dict(sd, strict=False)
            info["head_missing"] = [k for k in missing if k.startswith("head.")]
            info["other_missing"] = [k for k in missing if not k.startswith("head.")]
            info["unexpected"] = list(unexpected)
    else:
        raise ValueError(f"unknown kind: {kind}")

    return model, info


def build_param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float,
                       llrd: float | None = None):
    """head / backbone の 2 階層。llrd が指定された場合は ViT のブロック単位で層別 lr を付す。"""
    head_kw = ("classifier", ".head.", "head.weight", "head.bias",
               ".fc.weight", ".fc.bias", "fc.weight", "fc.bias")
    no_decay_kw = ("bias", "norm", "bn", "ls1.gamma", "ls2.gamma", "gamma", "cls_token",
                   "pos_embed", "mask_token", "register_tokens", "reg_token", "rope")

    groups = []
    n_blocks = len(model.blocks) if (llrd is not None and hasattr(model, "blocks")) else 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(k in name for k in no_decay_kw) else weight_decay
        is_head = any(k in name for k in head_kw)
        if llrd is None or n_blocks == 0:
            lr = lr_head if is_head else lr_backbone
        else:
            if name.startswith("head."):
                lr = lr_head
            elif name.startswith("blocks."):
                depth_from_top = (n_blocks - 1) - int(name.split(".")[1])
                lr = lr_backbone * (llrd ** depth_from_top)
            else:
                lr = lr_backbone * (llrd ** n_blocks)
        groups.append({"params": [p], "lr": lr, "weight_decay": wd, "_name": name})
    return groups


# ============================== 学習ループ ==============================

def cosine_lr(epoch: float, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr_schedule(optimizer, base_lrs, factor):
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, use_amp: bool):
    model.eval()
    correct = total = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullcontext()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1
    return correct / max(1, total), cm


def macro_recall_precision(cm):
    rec, prec = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        rec.append(tp / max(1, cm[i, :].sum()))
        prec.append(tp / max(1, cm[:, i].sum()))
    return float(np.mean(rec)), float(np.mean(prec))


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx,
                    lr_factor, base_lrs, use_amp, grad_clip=1.0, log_every=20,
                    teacher=None, kd_alpha=0.0, kd_T=2.0):
    """teacher を渡すと知識蒸留を行う (渡さなければ従来どおり CE のみ)。

    損失は L = alpha * T^2 * KL(student/T || teacher/T) + (1 - alpha) * CE。
    教師と生徒は同じ入力テンソルを見る (条件 A の統制と揃えるため)。
    """
    model.train()
    apply_lr_schedule(optimizer, base_lrs, lr_factor)
    running_loss = 0.0
    correct = total = nan_count = 0
    t0 = time.time()
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullcontext()

    for it, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(imgs)
            loss = criterion(logits, labels)
            if teacher is not None and kd_alpha > 0:
                with torch.no_grad():
                    t_logits = teacher(imgs).float()
                kd = nn.functional.kl_div(
                    nn.functional.log_softmax(logits.float() / kd_T, dim=1),
                    nn.functional.softmax(t_logits / kd_T, dim=1),
                    reduction="batchmean") * (kd_T ** 2)
                loss = kd_alpha * kd + (1.0 - kd_alpha) * loss
        if not torch.isfinite(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"  [nan] ep{epoch_idx} it{it} loss={loss.item()}", flush=True)
            continue
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            bs = imgs.size(0)
            running_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += bs
        if (it + 1) % log_every == 0:
            # LLRD 使用時は param_groups[0] が最下層になり極端に小さい値が出るため、最大 lr (head) を表示する
            lr_max = max(g["lr"] for g in optimizer.param_groups)
            print(f"  ep{epoch_idx} it{it + 1}/{len(loader)} loss={running_loss / total:.4f} "
                  f"acc={correct / total:.4f} lr_head={lr_max:.2e}", flush=True)
    return running_loss / max(1, total), correct / max(1, total), time.time() - t0, nan_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    p.add_argument("--res", type=int, required=True, help="解像度 N (情報量の定義)")
    p.add_argument("--mode", choices=["A", "B"], default="B",
                   help="A=224 へ復元して入力 / B=N をネイティブ入力 (既定)")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--splits_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pretrain_dir", type=Path, default=None, help="ViT-L 重みの置き場")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true", help="1 epoch で打ち切る")
    p.add_argument("--no_pretrained", action="store_true")
    # 既定はモデル別スペック。明示指定時のみ上書きする (感度分析用)
    for key in ("epochs", "batch_size", "warmup_epochs", "patience"):
        p.add_argument(f"--{key}", type=int, default=None)
    for key in ("lr_backbone", "lr_head"):
        p.add_argument(f"--{key}", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--no_amp", action="store_true")
    # ---- 知識蒸留 (既定 OFF。指定しなければ従来と完全に同じ挙動) ----
    p.add_argument("--teacher_model", choices=list(MODEL_SPECS), default=None,
                   help="蒸留の教師モデル (指定すると KD を有効化)")
    p.add_argument("--teacher_ckpt", type=Path, default=None, help="教師の best.pt")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="KD 項の重み")
    p.add_argument("--kd_T", type=float, default=2.0, help="温度")
    args = p.parse_args()

    if args.res not in RESOLUTIONS:
        print(f"[warn] res={args.res} は既定の 14 水準に含まれない (続行する)", flush=True)

    spec = dict(MODEL_SPECS[args.model])
    for key in ("epochs", "batch_size", "warmup_epochs", "lr_backbone", "lr_head"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    if args.patience is not None:
        spec["patience"] = args.patience
    use_amp = spec["amp"] and not args.no_amp

    input_res = resolve_input_res(args.model, args.res, args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[init] model={args.model} res={args.res} mode={args.mode} input_res={input_res} "
          f"device={device} amp={use_amp}", flush=True)
    if input_res != args.res:
        reason = "mode A の 224 復元" if args.mode == "A" else f"patch {spec['patch']} 倍数への丸め"
        print(f"[note] 入力寸法が解像度と異なる: {args.res} -> {input_res} ({reason})", flush=True)

    # ---- Data ----
    train_tf, val_tf = build_transforms(args.res, input_res)
    train_ds = SpeciesDataset(args.splits_dir / "train.csv", args.data_root, train_tf)
    val_ds = SpeciesDataset(args.splits_dir / "val.csv", args.data_root, val_tf)
    test_ds = SpeciesDataset(args.splits_dir / "test.csv", args.data_root, val_tf)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    train_labels = train_ds.df["species_idx"].values
    class_counts = np.bincount(train_labels, minlength=args.num_classes).astype(np.float64)
    sample_weights = (1.0 / np.maximum(class_counts, 1.0))[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=spec["batch_size"], sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=spec["batch_size"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=spec["batch_size"], shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    cls_w = torch.tensor((class_counts.sum() / args.num_classes) / np.maximum(class_counts, 1.0),
                         dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing).to(device)

    # ---- Model ----
    model, minfo = build_model(args.model, args.num_classes, input_res, args.pretrain_dir,
                               pretrained=not args.no_pretrained)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[model] {minfo.get('timm_name')} params={n_params / 1e6:.2f}M info={ {k: v for k, v in minfo.items() if k != 'unexpected'} }", flush=True)
    model.to(device)

    pg = build_param_groups(model, spec["lr_backbone"], spec["lr_head"],
                            args.weight_decay, spec["llrd"])
    optimizer = torch.optim.AdamW(pg, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    print(f"[opt] groups={len(pg)} lr_b={spec['lr_backbone']} lr_h={spec['lr_head']} "
          f"llrd={spec['llrd']} epochs={spec['epochs']} bs={spec['batch_size']}", flush=True)

    # ---- 教師 (KD を使う場合のみ) ----
    teacher = None
    if args.teacher_model is not None:
        if args.teacher_ckpt is None or not args.teacher_ckpt.exists():
            raise SystemExit(f"[NG] teacher_ckpt が無い: {args.teacher_ckpt}")
        # 教師は生徒と同じ入力寸法で読む (同じテンソルを両者に流すため)
        t_input_res = resolve_input_res(args.teacher_model, args.res, args.mode)
        if t_input_res != input_res:
            print(f"[warn] 教師の入力寸法 {t_input_res} が生徒の {input_res} と異なる", flush=True)
        teacher, tinfo = build_model(args.teacher_model, args.num_classes, t_input_res,
                                     args.pretrain_dir, pretrained=False)
        sd = torch.load(args.teacher_ckpt, map_location="cpu")
        # 本 PJ の保存形式は {"epoch","model_state","val_acc","meta"}。
        # ここを取り違えると**教師がランダム初期化のまま学習が進み**、KD の結論そのものが
        # 無意味になるので、キー名の候補を明示し、読めなければ即座に落とす。
        for key in ("model_state", "state_dict", "model"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        missing, unexpected = teacher.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            raise SystemExit(f"[NG] 教師の重みが読めていない (missing={len(missing)} "
                             f"unexpected={len(unexpected)})。先頭: {list(missing)[:3]}")
        teacher.to(device).eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f"[kd] teacher={args.teacher_model} ckpt={args.teacher_ckpt} "
              f"alpha={args.kd_alpha} T={args.kd_T} missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)

    meta = {
        "model": args.model, "res": args.res, "mode": args.mode, "input_res": input_res,
        "patch": spec["patch"], "params_M": round(n_params / 1e6, 3),
        "seed": args.seed, "spec": {k: v for k, v in spec.items() if k != "pretrain_file"},
        "model_info": {k: (v if not isinstance(v, list) else v[:5]) for k, v in minfo.items()},
        "amp": use_amp, "num_classes": args.num_classes,
        "kd": None if teacher is None else {
            "teacher": args.teacher_model, "ckpt": str(args.teacher_ckpt),
            "alpha": args.kd_alpha, "T": args.kd_T},
    }
    json.dump(meta, open(args.output / "meta.json", "w"), indent=2, ensure_ascii=False)

    # ---- Train ----
    best_val_acc, best_epoch, patience = 0.0, -1, 0
    log = []
    t_start = time.time()
    for epoch in range(spec["epochs"]):
        lr_factor = cosine_lr(epoch + 0.5, spec["epochs"], spec["warmup_epochs"])
        tr_loss, tr_acc, dt, nan_count = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, lr_factor, base_lrs, use_amp,
            teacher=teacher, kd_alpha=args.kd_alpha, kd_T=args.kd_T)
        val_acc, val_cm = evaluate(model, val_loader, device, args.num_classes, use_amp)
        macro_rec, macro_prec = macro_recall_precision(val_cm)
        print(f"[ep {epoch}/{spec['epochs']}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_acc={val_acc:.4f} macro_rec={macro_rec:.4f} dt={dt:.1f}s nan={nan_count}", flush=True)
        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc,
                    "macro_recall": macro_rec, "macro_precision": macro_prec,
                    "dt_s": dt, "nan_batches": nan_count})
        pd.DataFrame(log).to_csv(args.output / "train_log.csv", index=False)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience = val_acc, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc,
                        "meta": meta}, args.output / "best.pt")
            np.save(args.output / "best_val_cm.npy", val_cm)
            print(f"  [save] best.pt val_acc={val_acc:.4f}", flush=True)
        else:
            patience += 1
            if patience >= spec["patience"]:
                print(f"[early stop] epoch={epoch} best_epoch={best_epoch} "
                      f"best_val_acc={best_val_acc:.4f}", flush=True)
                break
        if args.dry_run:
            print("[dry_run] 1 epoch で打ち切り", flush=True)
            break

    train_sec = time.time() - t_start

    # ---- Test ----
    result = dict(meta)
    if (args.output / "best.pt").exists():
        ckpt = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        test_acc, test_cm = evaluate(model, test_loader, device, args.num_classes, use_amp)
        macro_rec, macro_prec = macro_recall_precision(test_cm)
        per_class_recall = [float(test_cm[i, i] / max(1, test_cm[i, :].sum()))
                            for i in range(args.num_classes)]
        np.save(args.output / "test_cm.npy", test_cm)
        print(f"[test] acc={test_acc:.4f} macro_rec={macro_rec:.4f} macro_prec={macro_prec:.4f}", flush=True)
        print(f"[test confusion]\n{test_cm}", flush=True)
        result.update({
            "test_acc": test_acc, "macro_recall": macro_rec, "macro_precision": macro_prec,
            "per_class_recall": per_class_recall,
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "epochs_run": len(log), "train_sec": round(train_sec, 1),
            "n_test": int(test_cm.sum()),
        })
    else:
        result.update({"error": "best.pt が生成されなかった", "train_sec": round(train_sec, 1)})

    json.dump(result, open(args.output / "test_metrics.json", "w"), indent=2, ensure_ascii=False)
    print(f"[done] {args.output}/test_metrics.json  train_sec={train_sec:.1f}", flush=True)


if __name__ == "__main__":
    main()
