"""Pre-train TextHead on SemEval-2020 Task 9 (SentiMix Hinglish, 14,000 samples), then fine-tune on Memotion 3.

Evaluates on Memotion 3 validation set (1,500 samples).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

import config
import data
import features
from models import TextHead
from train import FocalLoss, calibrate_task_a, compute_metrics, print_metrics


def main():
    device = features.get_device()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    print(f"Device: {device}")

    # 1. Load SentiMix data
    sm_train = np.load(config.CACHE_DIR / "sentimix_xlm_roberta_train.npz")
    sm_dev = np.load(config.CACHE_DIR / "sentimix_xlm_roberta_dev.npz")
    x_sm_tr = torch.from_numpy(sm_train["feats"]).float()
    y_sm_tr = torch.from_numpy(sm_train["labels"]).long()
    x_sm_va = torch.from_numpy(sm_dev["feats"]).float()
    y_sm_va = torch.from_numpy(sm_dev["labels"]).long()
    print(f"SentiMix Train: {len(y_sm_tr)} samples | Dev: {len(y_sm_va)} samples")

    # 2. Load Memotion 3 data
    s_tr = data.load_split("train", "A", "text", samples=False, balance=False)
    s_va = data.load_split("val", "A", "text", samples=False, balance=False)
    m3_x_tr = torch.from_numpy(features.text_features("train", s_tr.ids, s_tr.texts, device)).float()
    m3_y_tr = torch.from_numpy(s_tr.y).long()
    m3_x_va = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float()
    m3_y_va = torch.from_numpy(s_va.y).long()
    print(f"Memotion 3 Train: {len(m3_y_tr)} samples | Val: {len(m3_y_va)} samples")

    model = TextHead(n_out=3).to(device)

    # ===== PHASE 1: Pre-train on SentiMix (14k Hinglish tweets) =====
    print("\n" + "=" * 60)
    print("PHASE 1: Pre-training TextHead on SentiMix (14,000 Hinglish tweets)")
    print("=" * 60)

    loss_fn_sm = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    for ep in range(1, 16):
        model.train()
        n = len(y_sm_tr)
        order = torch.randperm(n)
        tl, correct = 0.0, 0
        for i in range(0, n, 64):
            idx = order[i:i + 64]
            xb = x_sm_tr[idx].to(device)
            yb = y_sm_tr[idx].to(device)
            logits = model(xb)
            loss = loss_fn_sm(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(x_sm_va.to(device)).cpu()
            v_pred = v_logits.argmax(1).numpy()
            dev_f1 = f1_score(y_sm_va.numpy(), v_pred, average="macro")

        print(f"[PT] Epoch {ep:>2}/15 | Train Loss: {tl/n:.4f} Acc: {correct/n:.4f} | SentiMix Dev Macro F1: {dev_f1:.4f}")

    print("Phase 1 Complete.")

    # ===== PHASE 2: Fine-tune on Memotion 3 (Hinglish Memes) =====
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning on Memotion 3 (7,000 samples)")
    print("=" * 60)

    classes, counts = torch.unique(m3_y_tr, return_counts=True)
    weights = float(len(m3_y_tr)) / (len(classes) * (counts.float() ** 0.75))
    weights = (weights / weights.mean()).to(device)
    loss_fn_m3 = FocalLoss(weight=weights, gamma=1.5, label_smoothing=0.08)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    best_score = -1.0
    best_state = None
    best_logits = None
    best_ep = 1
    patience = 12
    patience_ctr = 0

    t0 = time.time()
    for ep in range(1, 26):
        model.train()
        n = len(m3_y_tr)
        order = torch.randperm(n)
        tl, correct = 0.0, 0
        for i in range(0, n, 32):
            idx = order[i:i + 32]
            xb = m3_x_tr[idx].to(device)
            yb = m3_y_tr[idx].to(device)
            logits = model(xb)
            loss = loss_fn_m3(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(m3_x_va.to(device)).cpu()
            logits_np = v_logits.numpy()
            m = compute_metrics("A", m3_y_va.numpy(), logits_np)
            current_score = m["macro_f1"]

        improved = False
        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = logits_np.copy()
            best_ep = ep
            patience_ctr = 0
            improved = True
        else:
            patience_ctr += 1

        marker = " 🏆 (New Best!)" if improved else ""
        print(f"[FT] Epoch {ep:>2}/25 | Train Loss: {tl/n:.4f} Acc: {correct/n:.4f} | Memotion 3 Val Macro F1: {current_score:.4f}{marker}")

        if patience_ctr >= patience and ep >= 12:
            print(f"Early stopping at epoch {ep}.")
            break

    print(f"\nFine-tuning completed in {time.time() - t0:.1f}s")
    print(f">> Restored best checkpoint from Epoch {best_ep} (Macro F1 = {best_score:.4f})")

    model.load_state_dict(best_state)
    final_m = compute_metrics("A", m3_y_va.numpy(), best_logits)
    print("\n===== Memotion 3 Validation Results =====")
    print_metrics("A", final_m)

    calib_logits = calibrate_task_a(best_logits, tau=0.15)
    calib_m = compute_metrics("A", m3_y_va.numpy(), calib_logits)
    print("\n===== With Calibration (tau=0.15) =====")
    print_metrics("A", calib_m)


if __name__ == "__main__":
    main()
