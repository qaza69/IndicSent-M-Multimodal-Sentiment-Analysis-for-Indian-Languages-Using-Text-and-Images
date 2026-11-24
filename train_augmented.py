"""Pre-train classification head on Memotion 1 (English, 6991 samples) then fine-tune on Memotion 3 (Hinglish, 7000 samples).

Usage:
  .venv/bin/python train_augmented.py --model multimodal --task A
  .venv/bin/python train_augmented.py --model text --task A
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import config
os.environ.setdefault("MPLCONFIGDIR", str(config.CACHE_DIR / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

import data
import features
from models import build_head
from train import FocalLoss, calibrate_task_a, compute_metrics, print_metrics, save_plots


def parse_args():
    p = argparse.ArgumentParser(description="Augmented training: Memotion 1 pre-train → Memotion 3 fine-tune")
    p.add_argument("--model", choices=["text", "image", "multimodal"], required=True)
    p.add_argument("--task", choices=["A", "B"], default="A")
    p.add_argument("--backbone", choices=["clip", "vgg16"], default="clip")
    p.add_argument("--fusion", choices=["se", "clip"], default="se")
    p.add_argument("--pretrain-epochs", type=int, default=15)
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pretrain-lr", type=float, default=5e-4)
    p.add_argument("--finetune-lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_memotion1_features(model_type, task, device):
    """Load pre-extracted Memotion 1 features and labels."""
    labels = np.load(config.CACHE_DIR / "memotion1_labels.npz", allow_pickle=True)
    y_a = labels["y_a"]
    y_b = labels["y_b"]
    y = y_a if task == "A" else y_b

    xs = []
    if model_type in ("text", "multimodal"):
        z = np.load(config.CACHE_DIR / "memotion1_xlm_roberta.npz")
        xs.append(torch.from_numpy(z["feats"]))
    if model_type in ("image", "multimodal"):
        z = np.load(config.CACHE_DIR / "memotion1_clip.npz")
        xs.append(torch.from_numpy(z["feats"]))

    y_t = torch.from_numpy(y)
    print(f"[Memotion1] {len(y_t)} samples loaded")
    return xs, y_t


def load_memotion3_features(model_type, task, device):
    """Load Memotion 3 train + val features using existing pipeline."""
    s_tr = data.load_split("train", task, model_type, samples=False, balance=False)
    s_va = data.load_split("val", task, model_type, samples=False, balance=False)

    def get_feats(split, s):
        xs = []
        if model_type in ("text", "multimodal"):
            xs.append(features.text_features(split, s.ids, s.texts, device))
        if model_type in ("image", "multimodal"):
            xs.append(features.clip_features(split, s.ids, s.image_paths, device))
        return [torch.from_numpy(x) for x in xs], torch.from_numpy(s.y)

    x_tr, y_tr = get_feats("train", s_tr)
    x_va, y_va = get_feats("val", s_va)
    print(f"[Memotion3] train={len(y_tr)}, val={len(y_va)}")
    return x_tr, y_tr, x_va, y_va


def run_epoch(model, xs, y, batch_size, device, loss_fn, task, optimizer=None):
    training = optimizer is not None
    model.train(training)
    n = len(y)
    order = torch.randperm(n) if training else torch.arange(n)
    total_loss, correct, count = 0.0, 0.0, 0
    all_logits = []
    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = [x[idx].to(device).float() for x in xs]
            yb = y[idx].to(device)
            logits = model(*xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(idx)
            if task == "A":
                correct += (logits.argmax(1) == yb).float().sum().item()
                count += len(idx)
            else:
                correct += ((logits > 0) == (yb > 0.5)).float().sum().item()
                count += yb.numel()
            if not training:
                all_logits.append(logits.detach().cpu())
    out_logits = torch.cat(all_logits).numpy() if all_logits else None
    return total_loss / n, correct / count, out_logits


def make_loss(task, y_tr, device, gamma=1.5):
    if task == "A":
        classes, counts = torch.unique(y_tr, return_counts=True)
        weights = float(len(y_tr)) / (len(classes) * (counts.float() ** 0.75))
        weights = weights / weights.mean()
        return FocalLoss(weight=weights.to(device), gamma=gamma, label_smoothing=0.08)
    return nn.BCEWithLogitsLoss()


def main():
    args = parse_args()
    device = features.get_device()
    set_seed(args.seed)
    print(f"PyTorch {torch.__version__} | device: {device}")
    print(f"model={args.model} task={args.task} pretrain_epochs={args.pretrain_epochs} finetune_epochs={args.finetune_epochs}")

    # 1. Load all data
    x_m1, y_m1 = load_memotion1_features(args.model, args.task, device)
    x_tr, y_tr, x_va, y_va = load_memotion3_features(args.model, args.task, device)

    if args.task == "B":
        y_m1 = y_m1.float()
        y_tr = y_tr.float()
        y_va_t = y_va.float()
    else:
        y_va_t = y_va

    model = build_head(args.model, args.task, vision_backbone=args.backbone, fusion=args.fusion).to(device)

    # ========== PHASE 1: Pre-train on Memotion 1 ==========
    print(f"\n{'='*60}")
    print("PHASE 1: Pre-training on Memotion 1 ({} samples)".format(len(y_m1)))
    print('='*60)

    loss_fn = make_loss(args.task, y_m1, device, args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs, eta_min=1e-6)

    for ep in range(1, args.pretrain_epochs + 1):
        tl, ta, _ = run_epoch(model, x_m1, y_m1, args.batch_size, device, loss_fn, args.task, optimizer)
        _, _, logits = run_epoch(model, x_va, y_va_t, args.batch_size, device, loss_fn, args.task)
        scheduler.step()
        vm = compute_metrics(args.task, y_va.numpy(), logits)
        score = vm["macro_f1"] if args.task == "A" else vm["average"]["f1"]
        print(f"[PT] Epoch {ep:>3}/{args.pretrain_epochs}  loss={tl:.4f} acc={ta:.4f}  |  val Macro F1={score:.4f}")

    print("Pre-training done.")

    # ========== PHASE 2: Fine-tune on Memotion 3 ==========
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tuning on Memotion 3 ({} samples)".format(len(y_tr)))
    print('='*60)

    loss_fn = make_loss(args.task, y_tr, device, args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-6)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_score = -1.0
    best_state = None
    best_logits = None
    best_epoch = 1
    patience = 12
    patience_counter = 0
    metric_name = "Macro F1" if args.task == "A" else "Avg F1"

    t0 = time.time()
    for ep in range(1, args.finetune_epochs + 1):
        tl, ta, _ = run_epoch(model, x_tr, y_tr, args.batch_size, device, loss_fn, args.task, optimizer)
        vl, va, logits = run_epoch(model, x_va, y_va_t, args.batch_size, device, loss_fn, args.task)
        scheduler.step()

        vm = compute_metrics(args.task, y_va.numpy(), logits)
        current_score = vm["macro_f1"] if args.task == "A" else vm["average"]["f1"]

        for k, v in zip(["loss", "acc", "val_loss", "val_acc"], (tl, ta, vl, va)):
            history[k].append(v)
        history["val_f1"].append(current_score)

        improved = False
        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = logits.copy()
            best_epoch = ep
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"[FT] Epoch {ep:>3}/{args.finetune_epochs}  loss={tl:.4f} acc={ta:.4f}  |  val_loss={vl:.4f} val_acc={va:.4f}  {metric_name}={current_score:.4f}{marker}")

        if patience_counter >= patience and ep >= 12:
            print(f"Early stopping at epoch {ep} (no {metric_name} improvement for {patience} epochs).")
            break

    print(f"Fine-tuning time: {time.time() - t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        logits = best_logits
        print(f"\n>> Restored best checkpoint from Epoch {best_epoch} ({metric_name}={best_score:.4f})")

    metrics = compute_metrics(args.task, y_va.numpy(), logits)
    print_metrics(args.task, metrics)

    if args.task == "A" and args.calibrate:
        calib_logits = calibrate_task_a(logits, tau=args.tau)
        calib_m = compute_metrics(args.task, y_va.numpy(), calib_logits)
        print(f"\n===== Validation with Prior Calibration (tau={args.tau}) =====")
        print_metrics(args.task, calib_m)
        metrics["calibrated"] = calib_m

    tag = f"{args.model}_task{args.task}_augmented"
    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_plots(history, out_dir, tag)
    torch.save({
        "state_dict": model.state_dict(),
        "model": args.model,
        "task": args.task,
        "vision_backbone": args.backbone,
        "fusion": args.fusion if args.model == "multimodal" else None,
        "augmented": True,
    }, out_dir / "model.pt")
    print(f"\nSaved -> {out_dir}  (metrics.json, history.png, model.pt)")


if __name__ == "__main__":
    main()
