"""Multi-Task SEFusion with Scene-Augmented Text (BLIP + XLM-R + CLIP).

Combines:
- OCR text + Visual Scene description (768-dim XLM-RoBERTa)
- CLIP ViT-B/32 (512-dim visual features)
- Squeeze-and-Excitation modality gating
- Joint Task A (sentiment) + Task B (emotions) conditioning
"""
import time
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.special import softmax

import config
import data
import features
from models import MultiTaskSEFusionHead
from train import FocalLoss, calibrate_task_a, compute_metrics, print_metrics


def run_training(alpha=0.15, lr=2.5e-4, epochs=30, batch_size=32, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()
    print("=" * 65)
    print(f"MULTITASK SEFUSION (SCENE-AUGMENTED TEXT + CLIP) on {device}")
    print(f"Alpha: {alpha} | LR: {lr} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 65)

    # 1. Load Data
    s_tr_a = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_tr_b = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va_a = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    # Scene-augmented text features (768-dim)
    tr_txt_data = np.load(config.CACHE_DIR / "xlm_roberta_scene_train.npz")
    va_txt_data = np.load(config.CACHE_DIR / "xlm_roberta_scene_val.npz")
    tr_txt = torch.from_numpy(tr_txt_data["feats"]).float()
    va_txt = torch.from_numpy(va_txt_data["feats"]).float()

    # CLIP ViT-B/32 features (512-dim)
    tr_img = torch.from_numpy(features.clip_features("train", s_tr_a.ids, s_tr_a.image_paths, device)).float()
    va_img = torch.from_numpy(features.clip_features("val", s_va_a.ids, s_va_a.image_paths, device)).float()

    print(f"Features: Scene-Augmented Text={tr_txt.shape}, Vision={tr_img.shape}")

    y_tr_a = torch.from_numpy(s_tr_a.y).long()
    y_tr_b = torch.from_numpy(s_tr_b.y).float()
    y_va_a = torch.from_numpy(s_va_a.y).long()
    y_va_b = torch.from_numpy(s_va_b.y).float()

    model = MultiTaskSEFusionHead(vision_dim=features.CLIP_FEAT_DIM, hidden_dim=256).to(device)

    # Losses
    classes, counts = torch.unique(y_tr_a, return_counts=True)
    weights = float(len(y_tr_a)) / (len(classes) * (counts.float() ** 0.75))
    weights = (weights / weights.mean()).to(device)
    loss_fn_a = FocalLoss(weight=weights, gamma=1.5, label_smoothing=0.08)
    loss_fn_b = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score = -1.0
    best_state = None
    best_logits_a = None
    best_ep = 1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        n = len(y_tr_a)
        order = torch.randperm(n)
        tl, correct = 0.0, 0

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            tb = tr_txt[idx].to(device)
            ib = tr_img[idx].to(device)
            ya = y_tr_a[idx].to(device)
            yb = y_tr_b[idx].to(device)

            la_out, lb_out = model(tb, ib)
            la = loss_fn_a(la_out, ya)
            lb = loss_fn_b(lb_out, yb)
            loss = la + alpha * lb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * len(idx)
            correct += (la_out.argmax(1) == ya).sum().item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vla_out, vlb_out = model(va_txt.to(device), va_img.to(device))
            v_logits = vla_out.cpu().numpy()
            m = compute_metrics("A", y_va_a.numpy(), v_logits)
            score = m["macro_f1"]

        improved = score > best_score
        if improved:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits_a = v_logits.copy()
            best_ep = ep

        print(f"Epoch {ep:>2}/{epochs} | Loss: {tl/n:.4f} Acc: {correct/n:.4f} | Val Macro F1: {score:.4f}{' 🏆 NEW BEST' if improved else ''}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s")
    print(f">> Best Raw Epoch {best_ep} (Raw Macro F1: {best_score:.4f})")

    # Calibration sweep
    best_tau_score, best_tau = 0, 0
    for tau in np.linspace(0.02, 0.25, 24):
        calib = calibrate_task_a(best_logits_a, tau=tau)
        calib_mf1 = f1_score(y_va_a.numpy(), calib.argmax(1), average="macro")
        if calib_mf1 > best_tau_score:
            best_tau_score = calib_mf1
            best_tau = tau

    print(f">> Best Calibrated Macro F1 (tau={best_tau:.3f}): {best_tau_score:.4f}")
    calib_best = calibrate_task_a(best_logits_a, tau=best_tau)
    print_metrics("A", compute_metrics("A", y_va_a.numpy(), calib_best))

    out_dir = config.OUTPUT_DIR / "multitask_scene_augmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "model": "multitask_scene_augmented",
        "vision_backbone": "clip_vit_b32",
        "best_tau": best_tau,
        "raw_macro_f1": best_score,
        "calibrated_macro_f1": best_tau_score,
        "epoch": best_ep
    }, out_dir / "model.pt")
    np.savez_compressed(out_dir / "val_predictions.npz", logits=best_logits_a, y=y_va_a.numpy())
    print(f"Saved -> {out_dir / 'model.pt'}")
    return best_score, best_tau_score, best_logits_a, best_tau


if __name__ == "__main__":
    run_training()
