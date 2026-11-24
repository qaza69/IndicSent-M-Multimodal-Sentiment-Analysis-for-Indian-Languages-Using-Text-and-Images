"""Task A Optimization: Ordinal Distance Loss + Label Smoothing.

Sentiment classes in Memotion 3 are an ordered polarity scale:
0: Negative < 1: Neutral < 2: Positive

Standard Cross-Entropy treats all misclassifications equally.
This script introduces:
1. Label Smoothing (eps=0.10) to reduce over-confidence on noisy human annotations.
2. Ordinal Cost-Sensitive Loss: penalizes distance-2 errors (predicting Positive for Negative
   or vice-versa) 4x more than distance-1 errors (predicting Neutral).
3. Class-balanced prior weights to compensate for the val split shift.
4. Dual-Text (XLM-R + MuRIL = 1536) + CLIP ViT-B/32 (512) + Emotion Gating (4).
5. Post-hoc multi-class calibration.
"""
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

import config
import data
import features
from train_task_a_sota import DualTextSEFusion


class OrdinalSmoothingLoss(nn.Module):
    """Combines Label-Smoothed Cross-Entropy with an Ordinal Quadratic Distance Penalty."""

    def __init__(self, weight=None, label_smoothing=0.10, lambda_ordinal=0.35, num_classes=3):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.lambda_ordinal = lambda_ordinal
        self.num_classes = num_classes

        # Precompute cost matrix: C[i, j] = (i - j)^2
        # C[0, 2] = 4, C[0, 1] = 1, C[0, 0] = 0
        costs = np.zeros((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                costs[i, j] = float((i - j) ** 2)
        self.register_buffer("cost_matrix", torch.from_numpy(costs))

    def forward(self, logits, targets):
        # 1. Label-smoothed cross-entropy
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing
        )

        # 2. Ordinal expected penalty: sum_c ( (y - c)^2 * P_c )
        probs = F.softmax(logits, dim=1)  # [B, 3]
        batch_costs = self.cost_matrix[targets]  # [B, 3]
        ordinal_penalty = (probs * batch_costs).sum(dim=1).mean()

        # Combined loss
        return ce_loss + self.lambda_ordinal * ordinal_penalty


def train_ordinal_task_a(lr=2e-4, epochs=30, batch_size=32, seed=42, eps=0.10, lambda_ord=0.35):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()

    print("=" * 65)
    print(f"TRAINING TASK A WITH ORDINAL MARGIN LOSS ON {device}")
    print(f"Label Smoothing: eps={eps} | Ordinal Lambda: {lambda_ord} | LR: {lr}")
    print("=" * 65)

    s_tr = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_va = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_tr_b = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    xlm_tr = torch.from_numpy(features.text_features("train", s_tr.ids, s_tr.texts, device)).float()
    xlm_va = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float()
    muril_tr = torch.from_numpy(np.load(config.CACHE_DIR / "google_muril_base_cased_train.npz")["feats"]).float()
    muril_va = torch.from_numpy(np.load(config.CACHE_DIR / "google_muril_base_cased_val.npz")["feats"]).float()

    txt_tr = torch.cat([xlm_tr, muril_tr], dim=1)
    txt_va = torch.cat([xlm_va, muril_va], dim=1)

    img_tr = torch.from_numpy(features.clip_features("train", s_tr.ids, s_tr.image_paths, device)).float()
    img_va = torch.from_numpy(features.clip_features("val", s_va.ids, s_va.image_paths, device)).float()

    emo_tr = torch.from_numpy(s_tr_b.y).float()
    emo_va = torch.from_numpy(s_va_b.y).float()

    y_tr = torch.from_numpy(s_tr.y).long()
    y_va = s_va.y

    model = DualTextSEFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    weights = torch.tensor([1.25, 0.9, 1.05]).to(device)
    loss_fn = OrdinalSmoothingLoss(
        weight=weights,
        label_smoothing=eps,
        lambda_ordinal=lambda_ord,
        num_classes=3
    ).to(device)

    best_mf1 = 0.0
    best_state = None
    best_logits = None
    best_ep = 1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y_tr))
        tl = 0.0

        for i in range(0, len(y_tr), batch_size):
            idx = order[i:i + batch_size]
            tb = txt_tr[idx].to(device)
            ib = img_tr[idx].to(device)
            eb = emo_tr[idx].to(device)
            yb = y_tr[idx].to(device)

            logits = model(tb, ib, eb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(idx)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(txt_va.to(device), img_va.to(device), emo_va.to(device)).cpu().numpy()
            v_preds = v_logits.argmax(axis=1)
            mf1 = f1_score(y_va, v_preds, average="macro")
            wf1 = f1_score(y_va, v_preds, average="weighted")

        improved = mf1 > best_mf1
        if improved:
            best_mf1 = mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = v_logits.copy()
            best_ep = ep

        print(f"Ep {ep:>2}/{epochs} | Loss: {tl/len(y_tr):.4f} | Val Macro F1: {mf1*100:.2f}% | WF1: {wf1*100:.2f}%{' 🏆 NEW BEST' if improved else ''}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s | Best Ep: {best_ep} (Raw Macro F1: {best_mf1*100:.2f}%)")

    # Post-hoc Multi-Class Decision Threshold Calibration
    print("\n" + "=" * 65)
    print("CALIBRATING CLASS DECISION MULTIPLIERS (GRID SEARCH)")
    print("=" * 65)
    probs = torch.softmax(torch.from_numpy(best_logits), dim=1).numpy()
    best_cal_mf1 = best_mf1
    best_multipliers = [1.0, 1.0, 1.0]

    for m0 in np.linspace(0.8, 1.8, 11):
        for m1 in np.linspace(0.8, 1.3, 6):
            for m2 in np.linspace(0.8, 1.5, 8):
                preds = (probs * np.array([m0, m1, m2])).argmax(axis=1)
                mf1 = f1_score(y_va, preds, average="macro")
                if mf1 > best_cal_mf1:
                    best_cal_mf1 = mf1
                    best_multipliers = [float(m0), float(m1), float(m2)]

    final_preds = (probs * np.array(best_multipliers)).argmax(axis=1)
    final_macro = float(f1_score(y_va, final_preds, average="macro"))
    final_weighted = float(f1_score(y_va, final_preds, average="weighted"))
    final_acc = float(accuracy_score(y_va, final_preds))

    print("\n" + "=" * 65)
    print("FINAL OPTIMIZED VALIDATION REPORT (Task A - Ordinal Loss)")
    print("=" * 65)
    print(classification_report(y_va, final_preds, target_names=config.TASK_A_LABELS, digits=4))
    print(f"🏆 Final Calibrated Macro F1   : {final_macro*100:.2f}%")
    print(f"🏆 Final Calibrated Weighted F1 : {final_weighted*100:.2f}%")
    print(f"🏆 Final Calibrated Accuracy   : {final_acc*100:.2f}%")
    print(f"Optimal Class Multipliers     : {best_multipliers}")

    out_dir = config.OUTPUT_DIR / "multimodal_taskA_ordinal"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "model": "multimodal_taskA_ordinal",
        "macro_f1": final_macro,
        "weighted_f1": final_weighted,
        "accuracy": final_acc,
        "calibration_multipliers": best_multipliers,
        "epoch": best_ep,
        "label_smoothing": eps,
        "lambda_ordinal": lambda_ord
    }, out_dir / "model.pt")

    metrics = {
        "macro_f1": final_macro,
        "weighted_f1": final_weighted,
        "accuracy": final_acc,
        "multipliers": best_multipliers,
        "epoch": best_ep,
        "label_smoothing": eps,
        "lambda_ordinal": lambda_ord
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved checkpoint and metrics -> {out_dir}")
    return metrics


if __name__ == "__main__":
    train_ordinal_task_a()
