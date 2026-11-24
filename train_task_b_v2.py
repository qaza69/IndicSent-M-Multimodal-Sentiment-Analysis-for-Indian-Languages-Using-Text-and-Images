"""Task B V2 — Three improvements over V1:

1. BLIP Scene-Augmented Text (xlm_roberta_scene) instead of plain XLM-R.
   Scene captions add visual context that helps 'offensive' detection.

2. Focal Loss (gamma=2) instead of plain BCE so the model focuses hard on
   misclassified minority-class examples (especially offensive/motivational).

3. Label Smoothing (eps=0.05) on offensive & motivational to prevent
   over-confident bad predictions on noisy labels.

Saves to: outputs/multimodal_taskB_v2/model.pt
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


# ---------------------------------------------------------------------------
# Focal Loss (binary, element-wise)
# ---------------------------------------------------------------------------
class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss with per-label pos_weight and label smoothing."""

    def __init__(self, pos_weight, gamma=2.0, smooth=0.0):
        super().__init__()
        self.pos_weight = pos_weight   # shape [n_labels]
        self.gamma = gamma
        self.smooth = smooth           # label smoothing epsilon

    def forward(self, logits, targets):
        # Apply label smoothing
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth

        # Sigmoid probabilities
        p = torch.sigmoid(logits)
        # Standard BCE (unreduced)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none"
        )
        # Focal weight: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).mean()
        return loss


def run_training_task_b_v2(lr=2e-4, epochs=35, batch_size=32, seed=42,
                            gamma=2.0, smooth=0.05, pw_power=0.8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()
    print("=" * 65)
    print(f"TRAINING TASK B V2 (Focal + LabelSmooth + SceneText) ON {device}")
    print(f"LR: {lr} | Epochs: {epochs} | gamma: {gamma} | smooth: {smooth}")
    print("=" * 65)

    # 1. Load Data
    s_tr = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va = data.load_split("val",   "B", "multimodal", samples=False, balance=False)

    # --- Use BLIP scene-augmented XLM-R features ---
    # Scene-augmented XLM-R features (sequential ids 0..N-1 = dataset row index)
    scene_tr = np.load(config.CACHE_DIR / "xlm_roberta_scene_train.npz")
    scene_va = np.load(config.CACHE_DIR / "xlm_roberta_scene_val.npz")
    tr_feats_scene = scene_tr["feats"]  # shape [7000, 768]
    va_feats_scene = scene_va["feats"]  # shape [1500, 768]

    # s_tr.ids / s_va.ids are dataset row indices → direct numpy indexing
    tr_txt_scene = torch.from_numpy(tr_feats_scene[s_tr.ids]).float()
    va_txt_scene = torch.from_numpy(va_feats_scene[s_va.ids]).float()

    # Also keep plain XLM-R for fallback ids not in scene cache
    tr_txt_plain = torch.from_numpy(
        features.text_features("train", s_tr.ids, s_tr.texts, device)
    ).float()
    va_txt_plain = torch.from_numpy(
        features.text_features("val", s_va.ids, s_va.texts, device)
    ).float()

    # Fuse: average scene + plain XLM-R (both are 768-dim)
    tr_txt = 0.6 * tr_txt_scene + 0.4 * tr_txt_plain
    va_txt = 0.6 * va_txt_scene + 0.4 * va_txt_plain

    # CLIP image features
    tr_img = torch.from_numpy(
        features.clip_features("train", s_tr.ids, s_tr.image_paths, device)
    ).float()
    va_img = torch.from_numpy(
        features.clip_features("val", s_va.ids, s_va.image_paths, device)
    ).float()

    y_tr    = torch.from_numpy(s_tr.y).float()
    y_va_np = s_va.y

    # 2. Per-label pos_weight
    n_pos      = y_tr.sum(dim=0)
    n_neg      = float(len(y_tr)) - n_pos
    pos_weight = ((n_neg / n_pos) ** pw_power).to(device)
    print(f"pos_weight: {np.round(pos_weight.cpu().numpy(), 2)}")
    for j, name in enumerate(config.TASK_B_LABELS):
        pct = n_pos[j].item() / len(y_tr) * 100
        print(f"  {name:<14} | Train+: {int(n_pos[j].item())} ({pct:.1f}%) | w: {pos_weight[j].item():.2f}")

    loss_fn = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=gamma, smooth=smooth)

    # 3. Model — slightly wider hidden dim (384) for extra capacity
    model = SEFusionMultimodalHead(n_out=4, vision_dim=features.CLIP_FEAT_DIM, hidden_dim=384).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score  = -1.0
    best_state  = None
    best_logits = None
    best_ep     = 1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        n     = len(y_tr)
        order = torch.randperm(n)
        tl    = 0.0

        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            tb  = tr_txt[idx].to(device)
            ib  = tr_img[idx].to(device)
            yb  = y_tr[idx].to(device)

            logits = model(tb, ib)
            loss   = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(idx)

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(va_txt.to(device), va_img.to(device)).cpu().numpy()
        v_probs = expit(v_logits)
        macro_f1s   = [f1_score(y_va_np[:, j], (v_probs[:, j] > 0.5).astype(int),
                                average="macro") for j in range(4)]
        val_macro_avg = float(np.mean(macro_f1s))

        improved = val_macro_avg > best_score
        if improved:
            best_score  = val_macro_avg
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = v_logits.copy()
            best_ep     = ep

        tag = " 🏆 NEW BEST" if improved else ""
        print(f"Ep {ep:>2}/{epochs} | Loss: {tl/n:.4f} | Val AvgMacroF1: {val_macro_avg:.4f}{tag}")

    print(f"\nDone in {time.time()-t0:.1f}s | Best ep {best_ep} (Macro F1={best_score:.4f})")

    # 4. Optimal Threshold Search
    print("\n" + "=" * 65)
    print("OPTIMAL PER-EMOTION THRESHOLD SEARCH (V2 model)")
    print("=" * 65)
    best_probs    = expit(best_logits)
    opt_thresholds = []
    opt_preds      = []

    for j, name in enumerate(config.TASK_B_LABELS):
        best_th, best_th_f1 = 0.5, 0.0
        for th in np.linspace(0.05, 0.95, 91):
            pred_th = (best_probs[:, j] > th).astype(int)
            mf1 = f1_score(y_va_np[:, j], pred_th, average="macro")
            if mf1 > best_th_f1:
                best_th_f1 = mf1
                best_th    = th
        opt_thresholds.append(best_th)
        pred_col = (best_probs[:, j] > best_th).astype(int)
        opt_preds.append(pred_col)
        wf1 = f1_score(y_va_np[:, j], pred_col, average="weighted")
        acc = accuracy_score(y_va_np[:, j], pred_col)
        print(f"  {name:<14} | th={best_th:.2f} | WF1={wf1:.4f} | MacroF1={best_th_f1:.4f} | Acc={acc:.4f}")

    opt_pred_matrix = np.column_stack(opt_preds)
    final_avg_macro    = float(np.mean([f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="macro")    for j in range(4)]))
    final_avg_weighted = float(np.mean([f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted") for j in range(4)]))
    final_avg_acc      = float(np.mean([accuracy_score(y_va_np[:, j], opt_pred_matrix[:, j])               for j in range(4)]))

    print("\n" + "=" * 65)
    print("TASK B V2 FINAL (Standalone)")
    print(f"  Avg Macro F1   : {final_avg_macro:.4f}")
    print(f"  Avg Weighted F1: {final_avg_weighted:.4f}")
    print(f"  Avg Accuracy   : {final_avg_acc:.4f}")
    print("=" * 65)

    # 5. Save
    out_dir = config.OUTPUT_DIR / "multimodal_taskB_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict"       : model.state_dict(),
        "model"            : "multimodal_taskB_v2",
        "vision_backbone"  : "clip",
        "text_features"    : "xlm_roberta_scene_blend",
        "thresholds"       : opt_thresholds,
        "raw_macro_f1"     : best_score,
        "optimized_macro_f1"      : final_avg_macro,
        "optimized_weighted_f1"   : final_avg_weighted,
        "epoch"            : best_ep,
        "hyperparams"      : {"lr": lr, "epochs": epochs, "gamma": gamma,
                               "smooth": smooth, "pw_power": pw_power},
    }, out_dir / "model.pt")

    metrics_dict = {
        "average": {
            "macro_f1"   : final_avg_macro,
            "weighted_f1": final_avg_weighted,
            "accuracy"   : final_avg_acc,
        },
        "thresholds": {name: float(th) for name, th in zip(config.TASK_B_LABELS, opt_thresholds)},
        "per_label": {
            name: {
                "macro_f1"   : float(f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="macro")),
                "weighted_f1": float(f1_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted")),
                "accuracy"   : float(accuracy_score(y_va_np[:, j], opt_pred_matrix[:, j])),
                "precision"  : float(precision_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted", zero_division=0)),
                "recall"     : float(recall_score(y_va_np[:, j], opt_pred_matrix[:, j], average="weighted", zero_division=0)),
            }
            for j, name in enumerate(config.TASK_B_LABELS)
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved → {out_dir}")
    return metrics_dict


if __name__ == "__main__":
    run_training_task_b_v2()
