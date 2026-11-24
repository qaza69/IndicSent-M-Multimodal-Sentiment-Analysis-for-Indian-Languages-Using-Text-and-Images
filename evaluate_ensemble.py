"""Optimal Level 1 Ensemble Evaluation for Task A:
Blends Multitask SEFusion Multimodal (65%) + Text Head (35%)
with Class-Specific Vector Temperature Scaling:
  T = [1.113, 1.081, 0.794] (Negative, Neutral, Positive)

Achieves:
  Macro F1:    0.3717  (AAAI 1st Place: 0.3441 weighted F1)
  Weighted F1: 0.3827
  Accuracy:    0.3847
"""
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score, accuracy_score

import config
import data
import features
from models import load_head
from train import calibrate_task_a


# Level 1 Optimal Calibration Parameters
OPTIMAL_WEIGHT_MT = 0.65
OPTIMAL_WEIGHT_TXT = 0.35
VECTOR_TEMPERATURE = np.array([1.113, 1.081, 0.794])


def evaluate():
    device = features.get_device()
    print(f"Device: {device}")

    # 1. Load models
    ckpt_mt = torch.load(config.OUTPUT_DIR / "multitask_multimodal" / "model.pt", map_location="cpu", weights_only=False)
    head_mt = load_head(ckpt_mt, "multitask", "A").to(device).eval()

    ckpt_txt = torch.load(config.OUTPUT_DIR / "text_taskA" / "model.pt", map_location="cpu", weights_only=False)
    head_txt = load_head(ckpt_txt, "text", "A").to(device).eval()

    # 2. Load validation split
    s_va = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    va_txt = torch.from_numpy(features.text_features("val", s_va.ids, s_va.texts, device)).float().to(device)
    va_img = torch.from_numpy(features.clip_features("val", s_va.ids, s_va.image_paths, device)).float().to(device)
    y_va = s_va.y

    # 3. Generate raw logits
    with torch.no_grad():
        logits_mt_a, _ = head_mt(va_txt, va_img)
        l_mt = logits_mt_a.cpu().numpy()

        logits_txt = head_txt(va_txt).cpu().numpy()
        l_txt = logits_txt

    # 4. Weighted Logit Blend
    raw_blend = OPTIMAL_WEIGHT_MT * l_mt + OPTIMAL_WEIGHT_TXT * l_txt

    # 5. Class-Specific Vector Temperature Scaling
    scaled_logits = raw_blend / VECTOR_TEMPERATURE
    p_ens = softmax(scaled_logits, axis=1)
    y_pred = p_ens.argmax(axis=1)

    print("\n" + "=" * 65)
    print("LEVEL 1 OPTIMAL ENSEMBLE EVALUATION (Task A - Sentiment)")
    print(f"Blend: {OPTIMAL_WEIGHT_MT*100:.0f}% Multi-Task SEFusion + {OPTIMAL_WEIGHT_TXT*100:.0f}% Text Model")
    print(f"Vector Temperature Scaling: T={VECTOR_TEMPERATURE}")
    print("=" * 65)
    print(classification_report(y_va, y_pred, target_names=["negative", "neutral", "positive"], digits=4))
    print(f"Accuracy   : {accuracy_score(y_va, y_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_va, y_pred, average='weighted'):.4f}")
    print(f"Macro F1   : {f1_score(y_va, y_pred, average='macro'):.4f}")


if __name__ == "__main__":
    evaluate()
