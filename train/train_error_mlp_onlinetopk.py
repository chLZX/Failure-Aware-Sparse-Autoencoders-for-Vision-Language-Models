#!/usr/bin/env python3
import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from tasks.utils import get_sae_and_vit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAE_PATH = "/home/lzx/patchsae/data/sae_weight/base/out.pt"
BACKBONE = "openai/clip-vit-base-patch16"

DATASET_ROOTS = {
    "Imagenet1K": "/data1/lzx/Imagenet1K",
    "CIFAR10": "/data1/lzx/CIFAR/10",
    "CIFAR100": "/data1/lzx/CIFAR/100",
}

DOMAINS = ["ID", "OOD", "PAGD"]

INPUT_TYPE_BY_DOMAIN = {
    "ID": "pt",
    "OOD": "pt",
    "PAGD": "pt",
}

CLASS_NAMES = [
    "CLIP-correct",
    "CLIP-error-IDError",
    "CLIP-error-OODError",
    "CLIP-error-ADVError",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    path: str
    label: int
    dataset: str
    domain: str
    split: str
    set_name: str


class ImageListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if item.path.lower().endswith(".pt"):
            x = _load_pt_pixel_values(item.path)
        else:
            x = Image.open(item.path).convert("RGB")
        return x, item.label, item.path, item.dataset


def collate_batch(batch):
    images, labels, paths, datasets = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels, list(paths), list(datasets)


def _load_pt_pixel_values(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "pixel_values" in obj:
            obj = obj["pixel_values"]
        elif "image" in obj:
            obj = obj["image"]
        elif "img" in obj:
            obj = obj["img"]
        else:
            raise ValueError(f"Unsupported .pt dict keys: {list(obj.keys())[:20]}")
    if not torch.is_tensor(obj):
        raise ValueError(f"Unsupported .pt payload type: {type(obj)}")

    if obj.ndim == 4 and obj.shape[0] == 1:
        obj = obj[0]
    if obj.ndim == 3 and obj.shape[-1] == 3 and obj.shape[0] != 3:
        obj = obj.permute(2, 0, 1)
    if obj.ndim != 3 or obj.shape[0] != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(obj.shape)} from {path}")

    obj = obj.to(dtype=torch.float32)
    mx = obj.max().item()
    mn = obj.min().item()
    if mx > 5.0:
        obj = obj / 255.0
        mx = obj.max().item()
        mn = obj.min().item()
    if mn >= 0.0 and mx <= 1.5:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        obj = (obj - mean) / std

    return obj


def _batch_to_pixel_values(vit, images_or_tensors, device) -> torch.Tensor:
    bsz = len(images_or_tensors)
    pixel_values_by_idx: dict[int, torch.Tensor] = {}
    pil_images: list[object] = []
    pil_indices: list[int] = []

    for i, x in enumerate(images_or_tensors):
        if torch.is_tensor(x):
            pixel_values_by_idx[i] = x
        else:
            pil_images.append(x)
            pil_indices.append(i)

    if pil_images:
        pv = vit.processor(images=pil_images, return_tensors="pt", padding=True)["pixel_values"]
        for j, idx in enumerate(pil_indices):
            pixel_values_by_idx[idx] = pv[j]

    pixel_values = torch.stack([pixel_values_by_idx[i] for i in range(bsz)], dim=0).to(device)
    return pixel_values


def extract_sae_tokens_topk(vit, sae, cfg, images, device, topk: int) -> torch.Tensor:
    pixel_values = _batch_to_pixel_values(vit, images, device)
    text_inputs = vit.processor(text=[""] * pixel_values.shape[0], return_tensors="pt", padding=True).to(device)
    text_inputs["pixel_values"] = pixel_values
    hook_locations = [(cfg.block_layer, cfg.module_name)]
    _, vit_cache = vit.run_with_cache(hook_locations, **text_inputs)
    vit_act = vit_cache[(cfg.block_layer, cfg.module_name)]

    _, sae_cache = sae.run_with_cache(vit_act)
    sae_act = sae_cache["hook_hidden_post"]

    if sae_act.shape[0] != text_inputs["pixel_values"].shape[0]:
        sae_act = sae_act.permute(1, 0, 2)

    sae_tokens = sae_act[:, 1:197, :]
    mean_tokens = sae_tokens.mean(dim=1)  # (B, D)
    topk_vals, topk_idx = torch.topk(mean_tokens, k=topk, dim=-1)
    sparse = torch.zeros_like(mean_tokens)
    sparse.scatter_(dim=-1, index=topk_idx, src=topk_vals)
    return sparse

def list_inputs(base_dir, split_name, input_type: str):
    split_dir = Path(base_dir) / split_name
    if not split_dir.is_dir():
        return []
    out = []
    for p in split_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if input_type == "pt":
            if ext == ".pt":
                out.append(p)
        elif input_type == "image":
            if ext in IMG_EXTS:
                out.append(p)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
    return out


def build_samples(sample_frac: float, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    for dataset_name, root in DATASET_ROOTS.items():
        for domain in DOMAINS:
            base_dir = os.path.join(root, domain)
            if not os.path.isdir(base_dir):
                continue
            for split in ["fail", "success"]:
                if split == "success":
                    class_idx = CLASS_TO_IDX["CLIP-correct"]
                else:
                    if domain == "ID":
                        class_idx = CLASS_TO_IDX["CLIP-error-IDError"]
                    elif domain == "OOD":
                        class_idx = CLASS_TO_IDX["CLIP-error-OODError"]
                    elif domain == "PAGD":
                        class_idx = CLASS_TO_IDX["CLIP-error-ADVError"]
                    else:
                        raise ValueError(f"Unknown domain: {domain}")
                input_type = INPUT_TYPE_BY_DOMAIN[domain]
                group_paths = list_inputs(base_dir, split, input_type=input_type)
                if sample_frac < 1.0:
                    k = max(1, int(round(len(group_paths) * sample_frac))) if group_paths else 0
                    if k > 0:
                        group_paths = rng.sample(group_paths, k)
                for input_path in group_paths:
                    samples.append(Sample(str(input_path), class_idx, dataset_name, domain, split, "train"))
    return samples


def load_samples_from_split_file(path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                raise ValueError(f"Expected >=6 columns in split file, got {len(parts)}: {line[:200]}")
            pth, label_s, dataset, domain, split, set_name = parts[:6]
            samples.append(Sample(pth, int(label_s), dataset, domain, split, set_name))
    return samples


class ErrorClassifier(nn.Module):
    def __init__(self, topk: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        h1 = hidden_dim
        h2 = max(1, hidden_dim // 2)
        h3 = max(1, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(topk, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h3, 6),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(6, 4)
        )

    def forward(self, topk_vals: torch.Tensor) -> torch.Tensor:
        # topk_vals: (B, K)
        x = topk_vals.reshape(topk_vals.size(0), -1)
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_eval_epoch(model, vit, sae, cfg, loader, device, topk: int):
    total = 0
    correct = 0
    total_loss = 0.0
    model.eval()
    loop = tqdm(loader, desc="val")
    with torch.no_grad():
        for images, labels, _paths, _datasets in loop:
            topk_vals = extract_sae_tokens_topk(vit, sae, cfg, images, device, topk)
            labels = labels.to(device, non_blocking=True)
            logits = model(topk_vals)
            loss = nn.functional.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def run_eval_epoch_with_breakdown(model, vit, sae, cfg, loader, device, topk: int):
    total = 0
    correct = 0
    total_loss = 0.0
    per_ds = {}
    model.eval()
    loop = tqdm(loader, desc="test")
    with torch.no_grad():
        for images, labels, _paths, datasets in loop:
            topk_vals = extract_sae_tokens_topk(vit, sae, cfg, images, device, topk)
            labels = labels.to(device, non_blocking=True)
            logits = model(topk_vals)
            loss = nn.functional.cross_entropy(logits, labels, reduction="none")
            preds = logits.argmax(dim=1)

            bsz = labels.size(0)
            total += bsz
            correct += (preds == labels).sum().item()
            total_loss += loss.sum().item()

            for name in set(datasets):
                idxs = [i for i, d in enumerate(datasets) if d == name]
                if not idxs:
                    continue
                idxs_t = torch.tensor(idxs, device=labels.device)
                ds_total = len(idxs)
                ds_correct = (preds.index_select(0, idxs_t) == labels.index_select(0, idxs_t)).sum().item()
                ds_loss = loss.index_select(0, idxs_t).sum().item()
                stats = per_ds.setdefault(name, {"total": 0, "correct": 0, "loss": 0.0})
                stats["total"] += ds_total
                stats["correct"] += ds_correct
                stats["loss"] += ds_loss

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc, per_ds


def main():
    p = argparse.ArgumentParser(description="Train error detector MLP with online CLIP+SAE features (no feature saving).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--topk", type=int, default=300)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--split-file", type=str, default="/home/lzx/data/sampled_paths_with_split.txt")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--save-dir", type=str, default="/data1/lzx/ECCVtopk300+index/model_weight")
    p.add_argument("--log-dir", type=str, default="/data1/lzx/ECCVtopk300+index/model_output")
    args = p.parse_args()
    #print(-1)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = Path(args.log_dir) / "train_log.txt"
    #print(-2)
    sae, vit, cfg = get_sae_and_vit(
        sae_path=SAE_PATH,
        vit_type="base",
        device=str(device),
        backbone=BACKBONE,
        model_path=None,
        classnames=None,
    )
    for pp in sae.parameters():
        pp.requires_grad = False
    for pp in vit.model.parameters():
        pp.requires_grad = False
    vit.model.eval()
    #print(-5)
    samples = load_samples_from_split_file(args.split_file)
    #print(f"Total samples: {len(samples)}")
    #print(-7)
    samples_paths_txt = Path(args.log_dir) / "sampled_paths.txt"
    samples_counts_txt = Path(args.log_dir) / "sampled_counts.txt"
    counts: dict[tuple[str, str, int], int] = {}
    #print(0)
    with samples_paths_txt.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"{s.path}\t{s.label}\t{s.dataset}\t{s.domain}\t{s.split}\t{s.set_name}\n")
            key = (s.dataset, s.domain, s.label, s.set_name)
            counts[key] = counts.get(key, 0) + 1
    with samples_counts_txt.open("w", encoding="utf-8") as f:
        for (dataset, domain, label, set_name), cnt in sorted(counts.items()):
            f.write(f"{dataset}\t{domain}\tlabel_{label}\t{set_name}\t{cnt}\n")
    #print(000)
    train_samples = [s for s in samples if s.set_name == "train"]
    val_samples = [s for s in samples if s.set_name in {"valid", "val"}]
    test_samples = [s for s in samples if s.set_name == "test"]
    if not val_samples:
        raise ValueError("No valid/val samples found in split file; cannot select best model by valid acc.")
    train_ds = ImageListDataset(train_samples)
    val_ds = ImageListDataset(val_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_samples:
        test_ds = ImageListDataset(test_samples)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
        )

    model = ErrorClassifier(
        topk=49152,#args.topk,
        hidden_dim=args.hidden_dim,
        num_classes=len(CLASS_NAMES),
        dropout=args.dropout,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #print(1)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"start_time\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"batch_size\t{args.batch_size}\n")
        log_f.write(f"num_workers\t{args.num_workers}\n")
        log_f.flush()

        best_val_acc = -1.0
        best_path = None
        for epoch in range(1, args.epochs + 1):
            #print(11)
            epoch_t0 = time.time()
            model.train()
            train_total = 0
            train_correct = 0
            train_total_loss = 0.0
            window_total = 0
            window_correct = 0
            window_loss = 0.0

            loop = tqdm(train_loader, desc=f"train ep{epoch}")
            for step, (images, labels, _paths, _datasets) in enumerate(loop, start=1):
                #print(111)
                topk_vals = extract_sae_tokens_topk(vit, sae, cfg, images, device, args.topk)
                labels = labels.to(device, non_blocking=True)
                logits = model(topk_vals)
                loss = nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                bsz = labels.size(0)
                train_total += bsz
                train_correct += (preds == labels).sum().item()
                train_total_loss += loss.item() * bsz

                window_total += bsz
                window_correct += (preds == labels).sum().item()
                window_loss += loss.item() * bsz

                if args.log_every > 0 and step % args.log_every == 0:
                    w_loss = window_loss / max(1, window_total)
                    w_acc = window_correct / max(1, window_total)
                    msg = f"epoch {epoch} step {step} train_loss {w_loss:.4f} train_acc {w_acc:.3f}"
                    print(msg)
                    log_f.write(msg + "\n")
                    log_f.flush()
                    window_total = 0
                    window_correct = 0
                    window_loss = 0.0

            train_loss = train_total_loss / max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            val_loss, val_acc = run_eval_epoch(model, vit, sae, cfg, val_loader, device, args.topk)
            epoch_s = time.time() - epoch_t0

            msg = (
                f"epoch {epoch} end "
                f"train_loss {train_loss:.4f} train_acc {train_acc:.3f} "
                f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} "
                f"time_s {epoch_s:.1f}"
            )
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(args.save_dir, f"error_detector_best_epoch_{epoch:05d}.pt")
                torch.save(model.state_dict(), best_path)
                best_msg = f"best_update epoch {epoch} val_acc {val_acc:.4f} saved {best_path}"
                print(best_msg)
                log_f.write(best_msg + "\n")
                log_f.flush()

        if best_path and os.path.isfile(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            best_val_loss, best_val_acc = run_eval_epoch(model, vit, sae, cfg, val_loader, device, args.topk)
            msg = f"best_model valid_loss {best_val_loss:.4f} valid_acc {best_val_acc:.3f}"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()
            if test_loader is not None:
                test_loss, test_acc, test_by_ds = run_eval_epoch_with_breakdown(
                    model, vit, sae, cfg, test_loader, device, args.topk
                )
                msg = f"best_model test_loss {test_loss:.4f} test_acc {test_acc:.3f}"
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
                for name in sorted(test_by_ds.keys()):
                    stats = test_by_ds[name]
                    ds_acc = stats["correct"] / max(1, stats["total"])
                    ds_loss = stats["loss"] / max(1, stats["total"])
                    msg = (
                        f"best_model test_{name} "
                        f"loss {ds_loss:.4f} acc {ds_acc:.3f} "
                        f"n {stats['total']}"
                    )
                    print(msg)
                    log_f.write(msg + "\n")
                    log_f.flush()


if __name__ == "__main__":
    main()
