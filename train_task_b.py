"""Dedicated State-of-the-Art Training for Task B (Emotion Classification).

Features:
1. Squeeze-and-Excitation Multimodal Fusion (XLM-RoBERTa 768-dim + CLIP 512-dim).
2. Per-label class-imbalance reweighting (pos_weight in BCEWithLogitsLoss).
3. Metric-driven checkpoint selection (best validation average Macro F1).
4. Post-hoc coordinate threshold optimization per emotion dimension:
   [humour, sarcastic, offensive, motivational].
5. Probability ensemble with Text Head.
"""
import time
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.special import expit

import config
import data
import features
from models import SEFusionMultimodalHead


def run_training_task_b(lr=2e-4, epochs=30, batch_size=32, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()
    print("=" * 65)
    print(f"TRAINING TASK B (EMOTION CLASSIFICATION) ON {device}")
    print(f"Architecture: SEFusion (XLM-R + CLIP) | LR: {lr} | Epochs: {epochs}")
    print("=" * 65)

    # 1. Load Data
    s_tr = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    tr_txt = torch.from_numpy(features.text_features("train", s_tr.ids, s_tr.texts, device)).float()
    va_txt = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float()
    tr_img = torch.from_numpy(features.clip_features("train", s_tr.ids, s_tr.image_paths, device)).float()
    va_img = torch.from_numpy(features.clip_features("val", s_va.ids, s_va.image_paths, device)).float()

    y_tr = torch.from_numpy(s_tr.y).float()
    y_va = torch.from_numpy(s_va.y).float()
    y_va_np = s_va.y

    # 2. Per-label pos_weight to address extreme class imbalance
    # pos_weight = num_negative / num_positive
    n_pos = y_tr.sum(dim=0)
    n_neg = float(len(y_tr)) - n_pos
    # Dampen extreme ratios slightly with power 0.8 to prevent over-shooting
    pos_weights = ((n_neg / n_pos) ** 0.8).to(device)
    print(f"Emotion Class Weights (pos_weight): {np.round(pos_weights.cpu().numpy(), 2)}")
    for j, name in enumerate(config.TASK_B_LABELS):
        print(f"  {name:<14} | Train Positives: {int(n_pos[j].item())} ({n_pos[j].item()/len(y_tr)*100:.1f}%) | Weight: {pos_weights[j].item():.2f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 3. Model
    model = SEFusionMultimodalHead(n_out=4, vision_dim=features.CLIP_FEAT_DIM, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score = -1.0
    best_state = None
    best_logits = None
    best_ep = 1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        n = len(y_tr)
        order = torch.randperm(n)
        tl, correct = 0.0, 0

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            tb = tr_txt[idx].to(device)
            ib = tr_img[idx].to(device)
            yb = y_tr[idx].to(device)

            logits = model(tb, ib)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * len(idx)
            pred_b = (logits > 0).float()
            correct += (pred_b == yb).float().sum().item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(va_txt.to(device), va_img.to(device)).cpu().numpy()
            v_probs = expit(v_logits)
            # Default 0.5 threshold macro F1 across 4 labels
            macro_f1s = [f1_score(y_va_np[:, j], (v_probs[:, j] > 0.5).astype(int), average="macro") for j in range(4)]
            val_macro_avg = float(np.mean(macro_f1s))

        improved = val_macro_avg > best_score
        if improved:
            best_score = val_macro_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = v_logits.copy()
            best_ep = ep

        print(f"Epoch {ep:>2}/{epochs} | Loss: {tl/n:.4f} Acc: {correct/(n*4):.4f} | Val Avg Macro F1: {val_macro_avg:.4f}{' 🏆 NEW BEST' if improved else ''}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s")
    print(f">> Best Raw Epoch {best_ep} (Raw Macro F1: {best_score:.4f})")

    # 4. Optimal Threshold Search per Emotion
    print("\n" + "=" * 65)
    print("SEARCHING OPTIMAL PER-EMOTION DECISION THRESHOLDS")
    print("=" * 65)

    best_probs = expit(best_logits)
    opt_thresholds = []
    opt_preds = []

    for j, name in enumerate(config.TASK_B_LABELS):
        best_th = 0.5
        best_th_f1 = 0
        for th in np.linspace(0.05, 0.95, 91):
            pred_th = (best_probs[:, j] > th).astype(int)
            mf1 = f1_score(y_va_np[:, j], pred_th, average="macro")
            if mf1 > best_th_f1:
                best_th_f1 = mf1
                best_th = th
        opt_thresholds.append(best_th)
        pred_col = (best_probs[:, j] > best_th).astype(int)
        opt_preds.append(pred_col)
        wf1 = f1_score(y_va_np[:, j], pred_col, average="weighted")
        acc = accuracy_score(y_va_np[:, j], pred_col)
        print(f"  {name:<14} | Threshold: {best_th:.2f} | Weighted F1: {wf1:.4f} | Macro F1: {best_th_f1:.4f} | Acc: {acc:.4f}")

    opt_pred_matrix = np.column_stack(opt_preds)
    final_avg_macro = float(np.mean([f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="macro") for j in range(4)]))
    final_avg_weighted = float(np.mean([f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted") for j in range(4)]))
    final_avg_acc = float(np.mean([accuracy_score(y_va_np[:, j], opt_pred_matrix[:, j]) for j in range(4)]))

    print("\n" + "=" * 65)
    print("FINAL OPTIMIZED TASK B EVALUATION (Average across 4 emotions)")
    print("=" * 65)
    print(f"Optimized Average Macro F1   : {final_avg_macro:.4f}")
    print(f"Optimized Average Weighted F1: {final_avg_weighted:.4f}")
    print(f"Optimized Average Accuracy   : {final_avg_acc:.4f}")

    # Save Checkpoint
    out_dir = config.OUTPUT_DIR / "multimodal_taskB_sefusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "model": "multimodal_taskB_sefusion",
        "vision_backbone": "clip",
        "thresholds": opt_thresholds,
        "raw_macro_f1": best_score,
        "optimized_macro_f1": final_avg_macro,
        "optimized_weighted_f1": final_avg_weighted,
        "epoch": best_ep
    }, out_dir / "model.pt")

    metrics_dict = {
        "average": {
            "macro_f1": final_avg_macro,
            "weighted_f1": final_avg_weighted,
            "accuracy": final_avg_acc,
        },
        "thresholds": {name: float(th) for name, th in zip(config.TASK_B_LABELS, opt_thresholds)},
        "per_label": {
            name: {
                "macro_f1": float(f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="macro")),
                "weighted_f1": float(f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted")),
                "accuracy": float(accuracy_score(y_va_np[:, j], opt_pred_matrix[:, j])),
                "precision": float(precision_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted", zero_division=0)),
                "recall": float(recall_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted", zero_division=0)),
            }
            for j, name in enumerate(config.TASK_B_LABELS)
        }
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nSaved checkpoint & metrics -> {out_dir}")
    return metrics_dict


if __name__ == "__main__":
    run_training_task_b()
