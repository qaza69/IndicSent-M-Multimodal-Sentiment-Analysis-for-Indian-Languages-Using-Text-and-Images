"""Cross-Attention Multi-Task Head for Memotion 3 (Task A + Task B).

Architecture:
  - Bidirectional Cross-Attention:
      Text queries Vision (Q=t, K=v, V=v) -> t_cross
      Vision queries Text (Q=v, K=t, V=t) -> v_cross
  - Multi-Head Attention (4 heads, d_k=64, dim=256)
  - Residual connections & LayerNorm
  - Incongruity interaction: [t_cross, v_cross, |t_cross - v_cross|, t_cross * v_cross]
  - Multi-Task Conditioning:
      Task B emotion head predicts (humour, sarcasm, offensive, motivational)
      Task A sentiment head conditioned on shared cross-attention + emotion logits
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
from features import TEXT_FEAT_DIM, CLIP_FEAT_DIM
from train import FocalLoss, calibrate_task_a, compute_metrics, print_metrics


class CrossAttentionMultiTaskHead(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=4, dropout=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections
        self.txt_proj = nn.Sequential(
            nn.LayerNorm(TEXT_FEAT_DIM),
            nn.Dropout(dropout),
            nn.Linear(TEXT_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(CLIP_FEAT_DIM),
            nn.Dropout(dropout),
            nn.Linear(CLIP_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Bidirectional Cross-Attention
        self.mha_t2v = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.mha_v2t = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ln_t = nn.LayerNorm(hidden_dim)
        self.ln_v = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Shared interaction fusion: [t_cross, v_cross, diff, prod] = 4 * hidden_dim
        fusion_in_dim = hidden_dim * 4
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task B: Emotions (4 classes)
        self.head_b = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )

        # Task A: Sentiment (3 classes, conditioned on shared + emotion logits)
        self.head_a = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def forward(self, txt, img):
        # [B, hidden_dim] -> [B, 1, hidden_dim]
        t = self.txt_proj(txt).unsqueeze(1)
        v = self.img_proj(img).unsqueeze(1)

        # Text queries Vision
        t_attn, _ = self.mha_t2v(query=t, key=v, value=v)
        t_cross = self.ln_t((t + self.dropout(t_attn)).squeeze(1))

        # Vision queries Text
        v_attn, _ = self.mha_v2t(query=v, key=t, value=t)
        v_cross = self.ln_v((v + self.dropout(v_attn)).squeeze(1))

        # Incongruity and interaction vectors
        diff = torch.abs(t_cross - v_cross)
        prod = t_cross * v_cross

        fused = torch.cat([t_cross, v_cross, diff, prod], dim=1)
        shared_feat = self.shared_mlp(fused)

        logits_b = self.head_b(shared_feat)
        logits_a = self.head_a(torch.cat([shared_feat, logits_b], dim=1))
        return logits_a, logits_b


def run_training(alpha=0.15, lr=2.5e-4, epochs=30, batch_size=32, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = features.get_device()
    print(f"[Cross-Attention MultiTask] Device: {device} | Alpha: {alpha} | LR: {lr} | Epochs: {epochs}")

    # Load data
    s_tr_a = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_tr_b = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va_a = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    tr_txt = torch.from_numpy(features.text_features("train", s_tr_a.ids, s_tr_a.texts, device)).float()
    tr_img = torch.from_numpy(features.clip_features("train", s_tr_a.ids, s_tr_a.image_paths, device)).float()
    va_txt = torch.from_numpy(features.text_features("val", s_va_a.ids, s_va_a.texts, device)).float()
    va_img = torch.from_numpy(features.clip_features("val", s_va_a.ids, s_va_a.image_paths, device)).float()

    y_tr_a = torch.from_numpy(s_tr_a.y).long()
    y_tr_b = torch.from_numpy(s_tr_b.y).float()
    y_va_a = torch.from_numpy(s_va_a.y).long()
    y_va_b = torch.from_numpy(s_va_b.y).float()

    model = CrossAttentionMultiTaskHead().to(device)

    # Losses
    classes, counts = torch.unique(y_tr_a, return_counts=True)
    weights = float(len(y_tr_a)) / (len(classes) * (counts.float() ** 0.75))
    weights = (weights / weights.mean()).to(device)
    loss_fn_a = FocalLoss(weight=weights, gamma=1.5, label_smoothing=0.08)
    loss_fn_b = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score, best_state, best_logits_a = -1.0, None, None
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

        print(f"Epoch {ep:>2}/{epochs} | Loss: {tl/n:.4f} Acc: {correct/n:.4f} | Val Macro F1: {score:.4f}{' 🏆' if improved else ''}")

    print(f"\nTraining completed in {time.time() - t0:.1f}s")
    print(f">> Best Epoch {best_ep} (Raw Macro F1: {best_score:.4f})")

    # Calibration sweep
    best_tau_score, best_tau = 0, 0
    for tau in np.linspace(0.05, 0.25, 21):
        calib = calibrate_task_a(best_logits_a, tau=tau)
        calib_mf1 = f1_score(y_va_a.numpy(), calib.argmax(1), average="macro")
        if calib_mf1 > best_tau_score:
            best_tau_score = calib_mf1
            best_tau = tau

    print(f">> With Calibration (tau={best_tau:.3f}):")
    calib_best = calibrate_task_a(best_logits_a, tau=best_tau)
    print_metrics("A", compute_metrics("A", y_va_a.numpy(), calib_best))

    # Save checkpoint
    out_dir = config.OUTPUT_DIR / "multitask_crossattn"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "model": "crossattn_multitask",
        "vision_backbone": "clip",
        "fusion": "cross_attention",
        "best_tau": best_tau,
        "raw_macro_f1": best_score,
        "calibrated_macro_f1": best_tau_score,
    }, out_dir / "model.pt")
    print(f"Saved -> {out_dir / 'model.pt'}")
    return best_score, best_tau_score, best_logits_a, best_tau


if __name__ == "__main__":
    run_training()
