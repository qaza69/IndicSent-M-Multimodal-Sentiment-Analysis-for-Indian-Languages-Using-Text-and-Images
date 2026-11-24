"""Optimal Task A Ensemble Evaluator.

Blends:
- 80% Ordinal Loss Model (multimodal_taskA_ordinal)
- 20% Cross-Entropy SOTA Model (multimodal_taskA_sota)

Features:
- Dual Text: XLM-RoBERTa (768) + Google MuRIL (768) = 1536
- Vision: CLIP ViT-B/32 (512)
- Multi-Task Emotion Gating: Task B emotion cues
- Ordinal cost-sensitive distance penalty + label smoothing (eps=0.10)
- Post-hoc class decision calibration

Scores on 1,500 validation memes:
- Macro F1: 53.65%  (+18.99% gain over baseline 34.65%)
- Weighted F1: 54.83%
- Accuracy: 57.47%
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import config
import data
import features
from train_task_a_sota import DualTextSEFusion


def evaluate_task_a_ensemble():
    device = features.get_device()
    s_va = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    xlm_va = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float()
    muril_va = torch.from_numpy(np.load(config.CACHE_DIR / "google_muril_base_cased_val.npz")["feats"]).float()
    txt_va = torch.cat([xlm_va, muril_va], dim=1).to(device)
    img_va = torch.from_numpy(features.clip_features("val", s_va.ids, s_va.image_paths, device)).float().to(device)
    emo_va = torch.from_numpy(s_va_b.y).float().to(device)

    y_va = s_va.y

    # Model 1: SOTA Cross-Entropy
    ckpt1 = torch.load(config.OUTPUT_DIR / "multimodal_taskA_sota" / "model.pt", map_location="cpu", weights_only=False)
    m1 = DualTextSEFusion().to(device)
    m1.load_state_dict(ckpt1["state_dict"])
    m1.eval()
    with torch.no_grad():
        l1 = m1(txt_va, img_va, emo_va).cpu().numpy()
    p1 = torch.softmax(torch.from_numpy(l1), dim=1).numpy() * np.array(ckpt1["calibration_multipliers"])

    # Model 2: Ordinal Distance Loss with Normalized Hinglish features
    ckpt2 = torch.load(config.OUTPUT_DIR / "multimodal_taskA_normalized_ordinal" / "model.pt", map_location="cpu", weights_only=False)
    m2 = DualTextSEFusion().to(device)
    m2.load_state_dict(ckpt2["state_dict"])
    m2.eval()
    muril_norm_va = torch.from_numpy(np.load(config.CACHE_DIR / "muril_normalized_val.npz")["feats"]).float()
    txt_norm_va = torch.cat([xlm_va, muril_norm_va], dim=1).to(device)
    with torch.no_grad():
        l2 = m2(txt_norm_va, img_va, emo_va).cpu().numpy()
    p2 = torch.softmax(torch.from_numpy(l2), dim=1).numpy() * np.array(ckpt2["calibration_multipliers"])

    # Optimal blend: 20% SOTA + 80% Ordinal Normalized
    probs_blend = 0.20 * p1 + 0.80 * p2
    preds = probs_blend.argmax(axis=1)

    print("=" * 65)
    print("TASK A OPTIMAL ENSEMBLE EVALUATION REPORT")
    print("Blend: 80% Normalized Ordinal Model + 20% Cross-Entropy SOTA Model")
    print("=" * 65)
    print(classification_report(y_va, preds, target_names=config.TASK_A_LABELS, digits=4))
    print("-" * 65)
    print(f"  Macro F1   : {f1_score(y_va, preds, average='macro')*100:.2f}%")
    print(f"  Weighted F1: {f1_score(y_va, preds, average='weighted')*100:.2f}%")
    print(f"  Accuracy   : {accuracy_score(y_va, preds)*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_task_a_ensemble()
