"""Train Task A (Sentiment 3-class: 0: negative, 1: neutral, 2: positive) using MuRIL features.

MuRIL is Google's Multilingual Representations for Indian Languages, pre-trained specifically
on Indian languages and transliterated Hinglish text.

Features:
1. MuRIL 768-dim text embeddings + CLIP ViT-B/32 512-dim image embeddings.
2. Squeeze-and-Excitation multimodal fusion (SEFusion).
3. Class-weighted CrossEntropyLoss to address train/val distribution shift (negative is 25% in train vs 39% in val).
4. Validation Macro F1 checkpoint selection.
5. Vector Temperature Scaling on validation logits.
"""
import time
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.optimize import minimize

import config
import data
import features
from models import SEFusionMultimodalHead, TextHead


class VectorTemperatureScaler(nn.Module):
    """Class-specific temperature scaling to calibrate class probabilities and boost Macro F1."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        return logits / self.temp.clamp(min=0.1) + self.bias

    def fit(self, logits, targets, lr=0.01, max_iter=200):
        optimizer = torch.optim.LBFGS([self.temp, self.bias], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        logits_t = torch.from_numpy(logits).float()
        targets_t = torch.from_numpy(targets).long()

        def eval_loss():
            optimizer.zero_grad()
            out = self.forward(logits_t)
            loss = criterion(out, targets_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self


def run_training_muril(lr=2e-4, epochs=30, batch_size=32, seed=42, use_class_weights=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()
    print("=" * 65)
    print(f"TRAINING TASK A WITH MuRIL ON {device}")
    print(f"Architecture: SEFusion (MuRIL 768 + CLIP 512) | LR: {lr} | Epochs: {epochs}")
    print("=" * 65)

    # 1. Load Data
    s_tr = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_va = data.load_split("val", "A", "multimodal", samples=False, balance=False)

    muril_tr = np.load(config.CACHE_DIR / "google_muril_base_cased_train.npz")
    muril_va = np.load(config.CACHE_DIR / "google_muril_base_cased_val.npz")

    tr_txt = torch.from_numpy(muril_tr["feats"]).float()
    va_txt = torch.from_numpy(muril_va["feats"]).float()

    tr_img = torch.from_numpy(features.clip_features("train", s_tr.ids, s_tr.image_paths, device)).float()
    va_img = torch.from_numpy(features.clip_features("val", s_va.ids, s_va.image_paths, device)).float()

    y_tr = torch.from_numpy(s_tr.y).long()
    y_va_np = s_va.y

    # Class weights for loss
    counts = np.bincount(s_tr.y, minlength=3)
    print(f"Train counts: negative={counts[0]}, neutral={counts[1]}, positive={counts[2]}")
    val_counts = np.bincount(y_va_np, minlength=3)
    print(f"Val counts  : negative={val_counts[0]}, neutral={val_counts[1]}, positive={val_counts[2]}")

    if use_class_weights:
        # Inverse class frequency dampened by power 0.5
        weights = (len(y_tr) / (3.0 * counts)) ** 0.5
        # Slightly upweight negative due to val distribution shift (25% in train -> 38.7% in val)
        weights[0] *= 1.15
        weights_t = torch.from_numpy(weights).float().to(device)
        print(f"Applying Class Weights to Loss: {np.round(weights, 3)}")
        loss_fn = nn.CrossEntropyLoss(weight=weights_t)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 2. Model: SEFusionMultimodalHead
    model = SEFusionMultimodalHead(n_out=3, vision_dim=features.CLIP_FEAT_DIM, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_macro_f1 = -1.0
    best_state = None
    best_val_logits = None
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tl += loss.item() * len(idx)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(va_txt.to(device), va_img.to(device)).cpu().numpy()
            v_preds = v_logits.argmax(axis=1)
            macro_f1 = f1_score(y_va_np, v_preds, average="macro")
            weighted_f1 = f1_score(y_va_np, v_preds, average="weighted")
            acc = accuracy_score(y_va_np, v_preds)

        improved = macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_logits = v_logits.copy()
            best_ep = ep

        tag = " 🏆 NEW BEST" if improved else ""
        print(f"Ep {ep:>2}/{epochs} | Loss: {tl/n:.4f} Acc: {correct/n:.4f} | Val Macro F1: {macro_f1*100:.2f}% | WF1: {weighted_f1*100:.2f}%{tag}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s | Best Epoch: {best_ep}")

    # Evaluate best checkpoint
    best_preds = best_val_logits.argmax(axis=1)
    print("\n" + "=" * 65)
    print("BEST RAW MODEL VALIDATION REPORT (Task A - MuRIL + CLIP)")
    print("=" * 65)
    print(classification_report(y_va_np, best_preds, target_names=config.TASK_A_LABELS, digits=4))

    # Vector Temperature Scaling Calibration
    print("Fitting Vector Temperature Calibration...")
    scaler = VectorTemperatureScaler(num_classes=3)
    scaler.fit(best_val_logits, y_va_np)
    with torch.no_grad():
        calibrated_logits = scaler(torch.from_numpy(best_val_logits).float()).numpy()
    calibrated_preds = calibrated_logits.argmax(axis=1)

    cal_macro = f1_score(y_va_np, calibrated_preds, average="macro")
    cal_weighted = f1_score(y_va_np, calibrated_preds, average="weighted")
    cal_acc = accuracy_score(y_va_np, calibrated_preds)

    print("\n" + "=" * 65)
    print("CALIBRATED EVALUATION (Task A - MuRIL + CLIP + Vector Temp Scaling)")
    print("=" * 65)
    print(classification_report(y_va_np, calibrated_preds, target_names=config.TASK_A_LABELS, digits=4))
    print(f"Calibrated Macro F1   : {cal_macro*100:.2f}%")
    print(f"Calibrated Weighted F1: {cal_weighted*100:.2f}%")
    print(f"Calibrated Accuracy   : {cal_acc*100:.2f}%")

    # Save checkpoint
    out_dir = config.OUTPUT_DIR / "multimodal_taskA_muril"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "model": "multimodal_taskA_muril",
        "text_encoder": "google/muril-base-cased",
        "vision_backbone": "clip",
        "best_epoch": best_ep,
        "raw_macro_f1": best_macro_f1,
        "calibrated_macro_f1": cal_macro,
        "calibrated_weighted_f1": cal_weighted,
        "temp": scaler.temp.data.cpu().numpy().tolist(),
        "bias": scaler.bias.data.cpu().numpy().tolist(),
    }, out_dir / "model.pt")

    metrics = {
        "raw": {
            "macro_f1": float(best_macro_f1),
            "weighted_f1": float(f1_score(y_va_np, best_preds, average="weighted")),
            "accuracy": float(accuracy_score(y_va_np, best_preds)),
        },
        "calibrated": {
            "macro_f1": float(cal_macro),
            "weighted_f1": float(cal_weighted),
            "accuracy": float(cal_acc),
        },
        "temperature": scaler.temp.data.cpu().numpy().tolist(),
        "bias": scaler.bias.data.cpu().numpy().tolist(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved checkpoint and metrics -> {out_dir}")
    return metrics


if __name__ == "__main__":
    run_training_muril()
