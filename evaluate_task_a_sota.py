"""Evaluate State-of-the-Art Task A Model.

Architecture:
- Dual Text Encoders: XLM-RoBERTa (768) + Google MuRIL (768)
- Vision Encoder: CLIP ViT-B/32 (512)
- Multi-Task Emotion Conditioning: Task B emotion cues
- Squeeze-and-Excitation Modality Gating
- Class probability calibration

Usage:
    python evaluate_task_a_sota.py
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import config
import data
import features
from train_task_a_sota import DualTextSEFusion


def evaluate_task_a_sota():
    device = features.get_device()
    s_va = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    xlm_va = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float()
    muril_va = torch.from_numpy(np.load(config.CACHE_DIR / "google_muril_base_cased_val.npz")["feats"]).float()
    txt_va = torch.cat([xlm_va, muril_va], dim=1).to(device)
    img_va = torch.from_numpy(features.clip_features("val", s_va.ids, s_va.image_paths, device)).float().to(device)
    emo_va = torch.from_numpy(s_va_b.y).float().to(device)

    y_va = s_va.y

    ckpt_path = config.OUTPUT_DIR / "multimodal_taskA_sota" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = DualTextSEFusion().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(txt_va, img_va, emo_va).cpu().numpy()

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    multipliers = np.array(ckpt.get("calibration_multipliers", [1.0, 1.0, 1.0]))

    cal_preds = (probs * multipliers).argmax(axis=1)

    print("=" * 65)
    print("TASK A SOTA EVALUATION REPORT (Dual-Text + CLIP + Emotion Gated)")
    print("=" * 65)
    print(classification_report(y_va, cal_preds, target_names=config.TASK_A_LABELS, digits=4))
    print("-" * 65)
    print(f"  Macro F1   : {f1_score(y_va, cal_preds, average='macro')*100:.2f}%")
    print(f"  Weighted F1: {f1_score(y_va, cal_preds, average='weighted')*100:.2f}%")
    print(f"  Accuracy   : {accuracy_score(y_va, cal_preds)*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_task_a_sota()
