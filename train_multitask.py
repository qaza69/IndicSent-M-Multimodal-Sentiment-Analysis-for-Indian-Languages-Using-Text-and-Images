"""Joint Multi-Task Training: Task A (Sentiment) + Task B (Emotions) with SEFusion.

Architecture:
  - Text: XLM-RoBERTa (768)
  - Vision: CLIP ViT-B/32 (512)
  - Fusion: Squeeze-and-Excitation (SEFusion) dynamic gating
  - Task B Head: 4 binary emotion classifications (humour, sarcasm, offensive, motivational)
  - Task A Head: 3-class sentiment conditioned on shared features + emotion logits!

Loss:
  L_total = L_focal(Task A) + alpha * L_bce(Task B)

Usage:
  .venv/bin/python train_multitask.py
  .venv/bin/python train_multitask.py --alpha 0.4 --epochs 30 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

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
from models import build_head, MultiTaskSEFusionHead
from train import FocalLoss, calibrate_task_a, compute_metrics, print_metrics, save_plots


def parse_args():
    p = argparse.ArgumentParser(description="Joint Multi-Task Training (Task A + Task B)")
    p.add_argument("--alpha", type=float, default=0.15, help="Weight for Task B emotion loss (default: 0.15)")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay (default: 1e-2)")
    p.add_argument("--gamma", type=float, default=1.5, help="Focal loss gamma for Task A (default: 1.5)")
    p.add_argument("--tau", type=float, default=0.13, help="Calibration tau (default: 0.13)")
    p.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_multitask_data(device):
    """Load text and image features along with both Task A and Task B labels."""
    s_tr_a = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_tr_b = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va_a = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    print("Loading feature representations...")
    # Train features
    tr_txt = features.text_features("train", s_tr_a.ids, s_tr_a.texts, device)
    tr_img = features.clip_features("train", s_tr_a.ids, s_tr_a.image_paths, device)
    # Val features
    va_txt = features.text_features("val", s_va_a.ids, s_va_a.texts, device)
    va_img = features.clip_features("val", s_va_a.ids, s_va_a.image_paths, device)

    x_tr = [torch.from_numpy(tr_txt), torch.from_numpy(tr_img)]
    x_va = [torch.from_numpy(va_txt), torch.from_numpy(va_img)]

    y_tr_a = torch.from_numpy(s_tr_a.y)          # (7000,) int64
    y_tr_b = torch.from_numpy(s_tr_b.y).float()  # (7000, 4) float32
    y_va_a = torch.from_numpy(s_va_a.y)          # (1500,) int64
    y_va_b = torch.from_numpy(s_va_b.y).float()  # (1500, 4) float32

    print(f"Train samples: {len(y_tr_a)} | Val samples: {len(y_va_a)}")
    return x_tr, (y_tr_a, y_tr_b), x_va, (y_va_a, y_va_b)


def run_epoch(model, xs, ys, batch_size, device, loss_fn_a, loss_fn_b, alpha, optimizer=None):
    training = optimizer is not None
    model.train(training)
    y_a, y_b = ys
    n = len(y_a)
    order = torch.randperm(n) if training else torch.arange(n)

    total_loss, total_loss_a, total_loss_b = 0.0, 0.0, 0.0
    correct_a, count_a = 0, 0
    all_logits_a, all_logits_b = [], []

    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            txt_b = xs[0][idx].to(device).float()
            img_b = xs[1][idx].to(device).float()
            ya_b = y_a[idx].to(device)
            yb_b = y_b[idx].to(device)

            logits_a, logits_b = model(txt_b, img_b)

            la = loss_fn_a(logits_a, ya_b)
            lb = loss_fn_b(logits_b, yb_b)
            loss = la + alpha * lb

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = len(idx)
            total_loss += loss.item() * bs
            total_loss_a += la.item() * bs
            total_loss_b += lb.item() * bs

            correct_a += (logits_a.argmax(1) == ya_b).float().sum().item()
            count_a += bs

            if not training:
                all_logits_a.append(logits_a.detach().cpu())
                all_logits_b.append(logits_b.detach().cpu())

    out_logits_a = torch.cat(all_logits_a).numpy() if all_logits_a else None
    out_logits_b = torch.cat(all_logits_b).numpy() if all_logits_b else None

    return (
        total_loss / n,
        total_loss_a / n,
        total_loss_b / n,
        correct_a / count_a,
        (out_logits_a, out_logits_b)
    )


def main():
    args = parse_args()
    device = features.get_device()
    set_seed(args.seed)

    print("=" * 65)
    print("Multi-Task Learning: Task A (Sentiment) + Task B (Emotions)")
    print(f"Device: {device} | Alpha: {args.alpha} | LR: {args.lr} | Epochs: {args.epochs}")
    print("=" * 65)

    x_tr, (y_tr_a, y_tr_b), x_va, (y_va_a, y_va_b) = load_multitask_data(device)

    # Class-weighted Focal Loss for Task A
    classes, counts = torch.unique(y_tr_a, return_counts=True)
    weights = float(len(y_tr_a)) / (len(classes) * (counts.float() ** 0.75))
    weights = (weights / weights.mean()).to(device)
    loss_fn_a = FocalLoss(weight=weights, gamma=args.gamma, label_smoothing=0.08)

    # BCEWithLogitsLoss for Task B (multi-label emotions)
    loss_fn_b = nn.BCEWithLogitsLoss()

    model = MultiTaskSEFusionHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score = -1.0
    best_state = None
    best_logits_a = None
    best_logits_b = None
    best_epoch = 1
    patience_counter = 0

    history = {
        "loss": [], "loss_a": [], "loss_b": [], "acc_a": [],
        "val_loss": [], "val_loss_a": [], "val_loss_b": [], "val_acc_a": [],
        "val_macro_f1_a": [], "val_avg_f1_b": []
    }

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl, tla, tlb, ta_acc, _ = run_epoch(
            model, x_tr, (y_tr_a, y_tr_b), args.batch_size, device,
            loss_fn_a, loss_fn_b, args.alpha, optimizer
        )
        vl, vla, vlb, va_acc, (logits_a, logits_b) = run_epoch(
            model, x_va, (y_va_a, y_va_b), args.batch_size, device,
            loss_fn_a, loss_fn_b, args.alpha
        )
        scheduler.step()

        # Compute metrics
        m_a = compute_metrics("A", y_va_a.numpy(), logits_a)
        m_b = compute_metrics("B", y_va_b.numpy(), logits_b)

        current_score = m_a["macro_f1"]
        f1_b = m_b["average"]["f1"]

        history["loss"].append(tl)
        history["loss_a"].append(tla)
        history["loss_b"].append(tlb)
        history["acc_a"].append(ta_acc)
        history["val_loss"].append(vl)
        history["val_loss_a"].append(vla)
        history["val_loss_b"].append(vlb)
        history["val_acc_a"].append(va_acc)
        history["val_macro_f1_a"].append(current_score)
        history["val_avg_f1_b"].append(f1_b)

        improved = False
        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits_a = logits_a.copy()
            best_logits_b = logits_b.copy()
            best_epoch = ep
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        marker = " 🏆 (New Best!)" if improved else ""
        print(f"Epoch {ep:>2}/{args.epochs} | Loss: {tl:.4f} (A:{tla:.4f}, B:{tlb:.4f}) | Val Loss: {vl:.4f} | "
              f"Task A MF1: {current_score:.4f} | Task B F1: {f1_b:.4f}{marker}")

        if patience_counter >= args.patience and ep >= 12:
            print(f"Early stopping at epoch {ep} (no improvement for {args.patience} epochs).")
            break

    print(f"\nTraining completed in {time.time() - t0:.1f}s")
    print(f">> Restored best checkpoint from Epoch {best_epoch} (Task A Macro F1 = {best_score:.4f})")

    model.load_state_dict(best_state)

    print("\n" + "=" * 50)
    print("TASK A (Sentiment) Final Validation Results:")
    print("=" * 50)
    final_m_a = compute_metrics("A", y_va_a.numpy(), best_logits_a)
    print_metrics("A", final_m_a)

    if args.calibrate:
        calib_logits_a = calibrate_task_a(best_logits_a, tau=args.tau)
        calib_m_a = compute_metrics("A", y_va_a.numpy(), calib_logits_a)
        print(f"\n===== Task A Prior Calibration (tau={args.tau}) =====")
        print_metrics("A", calib_m_a)
        final_m_a["calibrated"] = calib_m_a

    print("\n" + "=" * 50)
    print("TASK B (Emotions) Final Validation Results:")
    print("=" * 50)
    final_m_b = compute_metrics("B", y_va_b.numpy(), best_logits_b)
    print_metrics("B", final_m_b)

    # Save outputs
    out_dir = config.OUTPUT_DIR / "multitask_multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        "task_A": final_m_a,
        "task_B": final_m_b,
        "best_epoch": best_epoch,
        "alpha": args.alpha,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({
        "state_dict": model.state_dict(),
        "model": "multitask",
        "vision_backbone": "clip",
        "fusion": "se_multitask",
    }, out_dir / "model.pt")

    print(f"\nCheckpoint successfully saved -> {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
