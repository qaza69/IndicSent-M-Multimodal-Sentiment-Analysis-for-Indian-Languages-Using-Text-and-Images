"""Task A Training with Hinglish-Normalized Text Features + Ordinal Smoothing Loss.

Combines:
1. Hinglish Normalized MuRIL features (768-dim).
2. XLM-RoBERTa multilingual features (768-dim).
3. CLIP ViT-B/32 image features (512-dim).
4. Multi-Task Emotion Gating (humour, sarcasm, offense, motivation).
5. Ordinal Quadratic Distance Cost Loss + Label Smoothing.
6. Post-hoc multi-class calibration.
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
from train_task_a_ordinal import OrdinalSmoothingLoss


def train_normalized_sota(lr=2e-4, epochs=30, batch_size=32, seed=42, eps=0.10, lambda_ord=0.35):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()

    print("=" * 65)
    print(f"TRAINING TASK A WITH HINGLISH-NORMALIZED FEATURES ON {device}")
    print(f"Architecture: XLM-R (768) + MuRIL-Normalized (768) + CLIP (512) + Emotion Gate")
    print(f"Loss: Ordinal Smoothing Loss (eps={eps}, lambda={lambda_ord})")
    print("=" * 65)

    s_tr = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_va = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_tr_b = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    xlm_tr = torch.from_numpy(features.text_features("train", s_tr.ids, s_tr.texts, device)).float()
    xlm_va = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float()

    muril_norm_tr = torch.from_numpy(np.load(config.CACHE_DIR / "muril_normalized_train.npz")["feats"]).float()
    muril_norm_va = torch.from_numpy(np.load(config.CACHE_DIR / "muril_normalized_val.npz")["feats"]).float()

    txt_tr = torch.cat([xlm_tr, muril_norm_tr], dim=1)
    txt_va = torch.cat([xlm_va, muril_norm_va], dim=1)

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

    # Post-hoc Multi-Class Decision Multipliers Search
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
    print("FINAL OPTIMIZED VALIDATION REPORT (Normalized Hinglish + Ordinal)")
    print("=" * 65)
    print(classification_report(y_va, final_preds, target_names=config.TASK_A_LABELS, digits=4))
    print(f"🏆 Final Calibrated Macro F1   : {final_macro*100:.2f}%")
    print(f"🏆 Final Calibrated Weighted F1 : {final_weighted*100:.2f}%")
    print(f"🏆 Final Calibrated Accuracy   : {final_acc*100:.2f}%")
    print(f"Optimal Class Multipliers     : {best_multipliers}")

    out_dir = config.OUTPUT_DIR / "multimodal_taskA_normalized_ordinal"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "model": "multimodal_taskA_normalized_ordinal",
        "macro_f1": final_macro,
        "weighted_f1": final_weighted,
        "accuracy": final_acc,
        "calibration_multipliers": best_multipliers,
        "epoch": best_ep
    }, out_dir / "model.pt")

    metrics = {
        "macro_f1": final_macro,
        "weighted_f1": final_weighted,
        "accuracy": final_acc,
        "multipliers": best_multipliers,
        "epoch": best_ep
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved checkpoint and metrics -> {out_dir}")
    return metrics


if __name__ == "__main__":
    train_normalized_sota()
