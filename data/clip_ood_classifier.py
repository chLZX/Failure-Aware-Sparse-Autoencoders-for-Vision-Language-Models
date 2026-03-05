#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except Exception:
    _HAS_OPEN_CLIP = False

try:
    import clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False


IMAGENET_CLASSNAMES = "/home/lzx/patchsae/configs/classnames/imagenet_classnames.txt"
CORRUPTIONS = {
    "brightness": "/home/lzx/data/imagenet_c/weather/brightness/1",
    "fog": "/home/lzx/data/imagenet_c/weather/fog/1",
    "frost": "/home/lzx/data/imagenet_c/weather/frost/1",
    "gaussian_noise": "/home/lzx/data/imagenet_c/noise/gaussian_noise/1",
    "impulse_noise": "/home/lzx/data/imagenet_c/noise/impulse_noise/1",
    "shot_noise": "/home/lzx/data/imagenet_c/noise/shot_noise/1",
    "contrast": "/home/lzx/data/imagenet_c/digital/contrast/1",
    "elastic_transform": "/home/lzx/data/imagenet_c/digital/elastic_transform/1",
    "jpeg_compression": "/home/lzx/data/imagenet_c/digital/jpeg_compression/1",
    "pixelate": "/home/lzx/data/imagenet_c/digital/pixelate/1",
    "defocus_blur": "/home/lzx/data/imagenet_c/blur/defocus_blur/1",
    "glass_blur": "/home/lzx/data/imagenet_c/blur/glass_blur/1",
    "motion_blur": "/home/lzx/data/imagenet_c/blur/motion_blur/1",
    "zoom_blur": "/home/lzx/data/imagenet_c/blur/zoom_blur/1",
    "snow": "/home/lzx/data/imagenet_c/weather/snow/1",
}
LABELS = None  # Use all ImageNet-1k labels from IMAGENET_CLASSNAMES.
OUT_ROOT = "/data1/lzx/d/dataset4SAEMLP/eval/imagenet/OOD"


def save_tensor(tensor, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(name).stem
    torch.save(tensor.detach().cpu(), os.path.join(output_dir, f"{stem}.pt"))


def load_classnames(path: str):
    synsets = []
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            synsets.append(parts[0])
            names.append(parts[1])
    return synsets, names


def load_clip(device: torch.device):
    if _HAS_OPEN_CLIP:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        model = model.to(device)
        return model, preprocess, tokenizer
    if _HAS_CLIP:
        model, preprocess = clip.load("ViT-B/16", device=device)
        tokenizer = clip.tokenize
        return model, preprocess, tokenizer
    raise RuntimeError("Neither open_clip nor clip is available.")


def iter_images(labels):
    items = []
    for corr_name, corr_root in CORRUPTIONS.items():
        for label in labels:
            label_dir = Path(corr_root) / label
            if not label_dir.is_dir():
                continue
            for img_path in label_dir.iterdir():
                if not img_path.is_file():
                    continue
                items.append((corr_name, label, img_path))
    return items


def main():
    os.makedirs(os.path.join(OUT_ROOT, "fail"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "success"), exist_ok=True)

    synsets, names = load_classnames(IMAGENET_CLASSNAMES)
    labels = synsets if LABELS is None else LABELS

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip(device)
    model.eval()

    prompts = [f"a photo of a {n}" for n in names]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    items = iter_images(labels)
    batch_size = 64

    total = 0
    correct = 0
    stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        images = []
        meta = []
        for corr_name, label, img_path in batch:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            images.append(preprocess(img))
            meta.append((corr_name, label, img_path.name))

        if not images:
            continue

        image_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1).cpu().tolist()

        for idx, ((corr_name, label, fname), pred_idx) in enumerate(zip(meta, preds)):
            pred_synset = synsets[pred_idx]
            is_correct = pred_synset == label
            total += 1
            correct += int(is_correct)
            key = (corr_name, label)
            stats[key]["total"] += 1
            stats[key]["correct"] += int(is_correct)
            split = "success" if is_correct else "fail"

            out_dir = os.path.join(OUT_ROOT, split, label)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{corr_name}_1_{fname}"
            src_path = os.path.join(CORRUPTIONS[corr_name], label, fname)
            if not os.path.exists(src_path):
                continue
            shutil.copy2(src_path, os.path.join(out_dir, out_name))
            save_tensor(images[idx], out_dir, out_name)

    print("Per-label per-corruption stats:")
    for corr_name in CORRUPTIONS.keys():
        for label in labels:
            key = (corr_name, label)
            if key not in stats:
                continue
            t = stats[key]["total"]
            c = stats[key]["correct"]
            acc = c / max(1, t)
            print(f"{corr_name}\t{label}\tTotal: {t}\tCorrect: {c}\tAcc: {acc:.3%}")

    acc = correct / max(1, total)
    print(f"Overall: Total: {total}  Correct: {correct}  Acc: {acc:.3%}")


if __name__ == "__main__":
    main()
