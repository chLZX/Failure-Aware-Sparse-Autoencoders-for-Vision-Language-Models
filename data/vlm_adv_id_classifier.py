import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.templates.openai_imagenet_templates import (  # noqa: E402
    openai_imagenet_template,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


class ClipClassifier(nn.Module):
    def __init__(self, model, text_features):
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        image_features = self.model.get_image_features(pixel_values=x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()


def _parse_imagenet_line(line: str):
    line = line.strip()
    if not line:
        return "", []
    if ":" in line:
        wnid, rest = line.split(":", 1)
        wnid = wnid.strip()
        rest = rest.strip()
    else:
        parts = line.split(maxsplit=1)
        wnid = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
    names = [n.strip() for n in rest.split(",") if n.strip()]
    return wnid, names


def load_imagenet_mapping(path):
    mapping = {}
    with open(path, "r") as file:
        for line in file:
            wnid, names = _parse_imagenet_line(line)
            if not wnid:
                continue
            if not names:
                names = [wnid]
            mapping[wnid] = names[0]
    return mapping


def load_imagenet_mapping_ordered(path):
    wnids = []
    primary_names = []
    synonyms = []
    mapping = {}
    with open(path, "r") as file:
        for line in file:
            wnid, names = _parse_imagenet_line(line)
            if not wnid:
                continue
            if not names:
                names = [wnid]
            wnids.append(wnid)
            primary_names.append(names[0])
            synonyms.append(names)
            mapping[wnid] = names[0]
    return wnids, primary_names, synonyms, mapping


def build_text_features(model, tokenizer, classnames, device):
    prompts = [f"a photo of a {c}" for c in classnames]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def save_tensor(tensor, output_dir, path):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(path).stem
    torch.save(tensor.detach().cpu(), os.path.join(output_dir, f"{stem}.pt"))


def pgd_linf_attack(model, images, labels, eps, alpha, steps):
    loss_fn = nn.CrossEntropyLoss()
    adv = images.detach().clone()
    adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = loss_fn(logits, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, images - eps), images + eps)
        adv = adv.clamp(0, 1)

    return adv.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/lzx/data/imagenet/imagenet_ori")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument(
        "--classnames",
        default="/home/lzx/patchsae/configs/classnames/imagenet_classnames.txt",
    )
    parser.add_argument(
        "--model-name", default="openai/clip-vit-base-patch16"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attack-batch-size", type=int, default=16)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root", default="/data1/lzx/generate_image"
    )
    parser.add_argument(
        "--log-path", default="/data1/lzx/generate_image/clip_adv.log"
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.split)
    transform = Compose([Resize(224), CenterCrop(224), ToTensor()])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    # Use all ImageNet1k classes in the split.
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.targets = [label for _, label in dataset.samples]

    wnids, classnames, classnames_synonyms, mapping = load_imagenet_mapping_ordered(args.classnames)
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    missing = [wnid for wnid in dataset.classes if wnid not in wnid_to_idx]
    if missing:
        raise ValueError(f"Missing WNIDs in mapping file: {missing[:5]}")
    wnid_to_name = {wnid: mapping.get(wnid, wnid) for wnid in dataset.classes}

    model = CLIPModel.from_pretrained(args.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model.eval().to(args.device)

    text_features = build_text_features(model, tokenizer, classnames, args.device)
    classifier = ClipClassifier(model, text_features).to(args.device)
    classifier.eval()

    pgd_success_root = os.path.join(args.output_root, "PGD", "success")
    pgd_fail_root = os.path.join(args.output_root, "PGD", "fail")
    os.makedirs(pgd_success_root, exist_ok=True)
    os.makedirs(pgd_fail_root, exist_ok=True)
    id_success_root = os.path.join(args.output_root, "ID", "success")
    id_fail_root = os.path.join(args.output_root, "ID", "fail")
    os.makedirs(id_success_root, exist_ok=True)
    os.makedirs(id_fail_root, exist_ok=True)

    for wnid in dataset.classes:
        os.makedirs(os.path.join(pgd_success_root, wnid), exist_ok=True)
        os.makedirs(os.path.join(pgd_fail_root, wnid), exist_ok=True)
        os.makedirs(os.path.join(id_success_root, wnid), exist_ok=True)
        os.makedirs(os.path.join(id_fail_root, wnid), exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    per_label_total = {wnid: 0 for wnid in dataset.classes}
    per_label_correct = {wnid: 0 for wnid in dataset.classes}
    per_label_attacked = {wnid: 0 for wnid in dataset.classes}
    per_label_adv_correct = {wnid: 0 for wnid in dataset.classes}
    adv_misclassified = 0
    adv_total = 0
    adv_target_logs = []

    label_map = torch.tensor(
        [wnid_to_idx[wnid] for wnid in dataset.classes], dtype=torch.long
    )

    for images, labels, paths in tqdm(dataloader, desc="Evaluating"):
        images = images.to(args.device)
        labels = labels.to(args.device)
        labels_1000 = label_map.to(args.device)[labels]
        with torch.no_grad():
            logits = classifier(images)
            preds = logits.argmax(dim=1)
        correct_mask = preds.eq(labels_1000)

        for idx, path in enumerate(paths):
            label = labels[idx].item()
            wnid = dataset.classes[label]
            per_label_total[wnid] += 1
            if correct_mask[idx].item():
                per_label_correct[wnid] += 1
                id_out_dir = os.path.join(id_success_root, wnid)
            else:
                id_out_dir = os.path.join(id_fail_root, wnid)
            save_tensor(images[idx], id_out_dir, path)

        if correct_mask.any():
            correct_indices = correct_mask.nonzero(as_tuple=False).squeeze(1)
            clean_correct = images[correct_indices]
            labels_correct = labels_1000[correct_indices]
            paths_correct = [paths[i] for i in correct_indices.tolist()]
            wnids_correct = [
                dataset.classes[labels[i].item()] for i in correct_indices.tolist()
            ]

            for wnid in wnids_correct:
                per_label_attacked[wnid] += 1
            adv_total += labels_correct.size(0)

            adv_images = pgd_linf_attack(
                classifier,
                clean_correct,
                labels_correct,
                eps=args.eps,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
            )
            with torch.no_grad():
                adv_logits = classifier(adv_images.to(args.device))
                adv_preds = adv_logits.argmax(dim=1)

            for idx, adv in enumerate(adv_images):
                true_label = labels_correct[idx].item()
                pred_label = adv_preds[idx].item()
                wnid = wnids_correct[idx]
                pred_wnid = wnids[pred_label] if 0 <= pred_label < len(wnids) else None
                if pred_label == true_label:
                    per_label_adv_correct[wnid] += 1
                    out_dir = os.path.join(pgd_success_root, wnid)
                else:
                    adv_misclassified += 1
                    out_dir = os.path.join(pgd_fail_root, wnid)
                    pred_name = wnid_to_name.get(pred_wnid, str(pred_label))
                    adv_target_logs.append(
                        f"{paths_correct[idx]}\t{wnid_to_name[wnid]}\t{pred_label}\t{pred_name}"
                    )
                save_tensor(adv, out_dir, paths_correct[idx])

    with open(args.log_path, "w") as log_file:
        log_file.write("Clean accuracy per label:\n")
        for wnid in dataset.classes:
            total = per_label_total[wnid]
            correct = per_label_correct[wnid]
            acc = correct / total if total else 0.0
            log_file.write(
                f"{wnid_to_name[wnid]}\t{acc:.4f}\t{correct}/{total}\n"
            )

        log_file.write("\nPGD accuracy per label (attacked subset only):\n")
        for wnid in dataset.classes:
            total = per_label_attacked[wnid]
            correct = per_label_adv_correct[wnid]
            acc = correct / total if total else 0.0
            log_file.write(
                f"{wnid_to_name[wnid]}\t{acc:.4f}\t{correct}/{total}\n"
            )

        adv_acc = (adv_total - adv_misclassified) / adv_total if adv_total else 0.0
        adv_success_rate = adv_misclassified / adv_total if adv_total else 0.0
        log_file.write("\nOverall PGD results (attacked subset only):\n")
        log_file.write(f"adv_accuracy\t{adv_acc:.4f}\n")
        log_file.write(f"adv_success_rate\t{adv_success_rate:.4f}\n")

        log_file.write("\nAttack success target labels:\n")
        log_file.write("image_path\ttrue_label\tadv_pred_label\tadv_pred_name\n")
        for line in adv_target_logs:
            log_file.write(f"{line}\n")


if __name__ == "__main__":
    main()
