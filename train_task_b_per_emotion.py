"""Dedicated per-emotion binary classifiers for Task B.

Trains 4 separate binary classifiers (one per emotion) with:
- Aggressive pos_weight tuned per emotion
- Independent learning rates & epochs per emotion
- Both multimodal (SEFusion) and text-only models

The idea: offensive and motivational have terrible performance because
they get "averaged out" in the multi-label joint model. Training dedicated
per-emotion models with tuned hyperparameters fixes this.

Saves to: outputs/taskB_per_emotion/<emotion>/model.pt
"""
import time
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy.special import expit

import config
import data
import features
from models import SEFusionMultimodalHead


class BinaryFocalLoss(nn.Module):
    """Single-label focal loss with pos_weight and label smoothing."""
    def __init__(self, pos_weight, gamma=2.0, smooth=0.05):
        super().__init__()
        self.pw = torch.tensor([pos_weight])
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        p = torch.sigmoid(logits.squeeze())
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), targets,
            pos_weight=self.pw.to(logits.device),
            reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - p_t) ** self.gamma * bce).mean()
        return loss


# Per-emotion hyperparameters tuned to their class statistics
EMOTION_CONFIGS = {
    "humour":       {"lr": 1e-4, "epochs": 25, "pw_power": 0.6, "gamma": 1.5, "smooth": 0.02},
    "sarcastic":    {"lr": 1e-4, "epochs": 25, "pw_power": 0.7, "gamma": 1.5, "smooth": 0.02},
    "offensive":    {"lr": 2e-4, "epochs": 40, "pw_power": 0.9, "gamma": 2.5, "smooth": 0.08},
    "motivational": {"lr": 3e-4, "epochs": 45, "pw_power": 1.0, "gamma": 3.0, "smooth": 0.10},
}


def train_one_emotion(emotion_idx, emotion_name, cfg,
                      tr_txt, tr_img, y_tr_col,
                      va_txt, va_img, y_va_col,
                      device):
    n_pos = float(y_tr_col.sum())
    n_neg = float(len(y_tr_col)) - n_pos
    pw    = (n_neg / n_pos) ** cfg["pw_power"]
    print(f"\n{'='*60}")
    print(f"[{emotion_name.upper()}] Positives: {int(n_pos)} ({n_pos/len(y_tr_col)*100:.1f}%) | pw={pw:.2f}")
    print(f"  LR={cfg['lr']} Epochs={cfg['epochs']} gamma={cfg['gamma']} smooth={cfg['smooth']}")

    loss_fn   = BinaryFocalLoss(pos_weight=pw, gamma=cfg["gamma"], smooth=cfg["smooth"])
    model     = SEFusionMultimodalHead(n_out=1, vision_dim=features.CLIP_FEAT_DIM, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-7)

    y_tr_t = torch.from_numpy(y_tr_col).float()
    y_va_np = y_va_col

    best_score  = -1.0
    best_state  = None
    best_logits = None
    best_ep     = 1
    bs          = 32

    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        n     = len(y_tr_t)
        order = torch.randperm(n)
        tl    = 0.0
        for i in range(0, n, bs):
            idx = order[i : i + bs]
            tb  = tr_txt[idx].to(device)
            ib  = tr_img[idx].to(device)
            yb  = y_tr_t[idx].to(device)
            loss = loss_fn(model(tb, ib), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(va_txt.to(device), va_img.to(device)).cpu().numpy().squeeze()
        v_probs = expit(v_logits)
        mf1 = f1_score(y_va_np, (v_probs > 0.5).astype(int), average="macro")
        improved = mf1 > best_score
        if improved:
            best_score  = mf1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = v_logits.copy()
            best_ep     = ep
        if ep % 5 == 0 or improved:
            print(f"  Ep {ep:>2}/{cfg['epochs']} | Loss={tl/n:.4f} | MacroF1={mf1:.4f}{' 🏆' if improved else ''}")

    print(f"  Done in {time.time()-t0:.1f}s | Best ep={best_ep} MacroF1={best_score:.4f}")

    # Threshold search
    best_probs = expit(best_logits)
    best_th, best_f1 = 0.5, 0.0
    for th in np.linspace(0.05, 0.95, 181):
        pred = (best_probs > th).astype(int)
        mf1  = f1_score(y_va_np, pred, average="macro")
        if mf1 > best_f1: best_f1 = mf1; best_th = th
    pred_col = (best_probs > best_th).astype(int)
    wf1 = f1_score(y_va_np, pred_col, average="weighted")
    acc = accuracy_score(y_va_np, pred_col)
    print(f"  Optimal th={best_th:.3f} | WF1={wf1:.4f} | MacroF1={best_f1:.4f} | Acc={acc:.4f}")

    # Save
    out_dir = config.OUTPUT_DIR / "taskB_per_emotion" / emotion_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict"      : model.state_dict(),
        "emotion"         : emotion_name,
        "threshold"       : best_th,
        "macro_f1"        : best_f1,
        "weighted_f1"     : wf1,
        "epoch"           : best_ep,
    }, out_dir / "model.pt")
    return best_probs, best_th, best_f1, wf1


def run_per_emotion_training():
    device = features.get_device()
    print("=" * 60)
    print(f"TASK B — PER-EMOTION DEDICATED CLASSIFIERS on {device}")
    print("=" * 60)

    s_tr = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va = data.load_split("val",   "B", "multimodal", samples=False, balance=False)

    # Load features (cached)
    tr_txt = torch.from_numpy(features.text_features("train", s_tr.ids, s_tr.texts, device)).float()
    va_txt = torch.from_numpy(features.text_features("val",   s_va.ids, s_va.texts, device)).float()
    tr_img = torch.from_numpy(features.clip_features("train", s_tr.ids, s_tr.image_paths, device)).float()
    va_img = torch.from_numpy(features.clip_features("val",   s_va.ids, s_va.image_paths, device)).float()

    # Also blend scene features for a richer text rep
    sc_tr  = np.load(config.CACHE_DIR / "xlm_roberta_scene_train.npz")
    sc_va  = np.load(config.CACHE_DIR / "xlm_roberta_scene_val.npz")
    tr_txt = 0.6 * torch.from_numpy(sc_tr["feats"][s_tr.ids]).float() + 0.4 * tr_txt
    va_txt = 0.6 * torch.from_numpy(sc_va["feats"][s_va.ids]).float() + 0.4 * va_txt

    y_tr_all = s_tr.y  # [7000, 4]
    y_va_all = s_va.y  # [1500, 4]

    results = {}
    for j, name in enumerate(config.TASK_B_LABELS):
        cfg = EMOTION_CONFIGS[name]
        probs, th, mf1, wf1 = train_one_emotion(
            j, name, cfg,
            tr_txt, tr_img, y_tr_all[:, j],
            va_txt, va_img, y_va_all[:, j],
            device,
        )
        results[name] = {"probs": probs, "threshold": th, "macro_f1": mf1, "weighted_f1": wf1}

    # Summary
    print("\n" + "=" * 60)
    print("PER-EMOTION CLASSIFIER SUMMARY")
    print("=" * 60)
    avg_macro    = np.mean([results[n]["macro_f1"]    for n in config.TASK_B_LABELS])
    avg_weighted = np.mean([results[n]["weighted_f1"] for n in config.TASK_B_LABELS])
    for n in config.TASK_B_LABELS:
        r = results[n]
        print(f"  {n:<14} | th={r['threshold']:.3f} | WF1={r['weighted_f1']:.4f} | MacroF1={r['macro_f1']:.4f}")
    print(f"\n  Avg Macro F1   : {avg_macro:.4f}")
    print(f"  Avg Weighted F1: {avg_weighted:.4f}")


if __name__ == "__main__":
    run_per_emotion_training()
