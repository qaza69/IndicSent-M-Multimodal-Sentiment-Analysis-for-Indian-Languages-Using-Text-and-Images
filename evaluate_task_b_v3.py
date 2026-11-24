"""Task B Final Ensemble Evaluator.

Blends three complementary models:
  A: multimodal_taskB_sefusion  (plain XLM-R + CLIP, BCE+pos_weight)
  B: multimodal_taskB_v2        (scene XLM-R blend + CLIP, Focal+LabelSmooth)
  C: text_taskB                 (plain XLM-R text-only)

Grid-searches blend weights (alpha_A, alpha_B, alpha_C) over a coarse grid,
then per-emotion threshold-tunes the winning blend.

Usage:
    python evaluate_task_b_v3.py
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.special import expit
from itertools import product

import config
import data
import features
from models import load_head, SEFusionMultimodalHead


def _load_probs_se(tag, n_out, device, va_txt, va_img):
    ckpt = torch.load(config.OUTPUT_DIR / tag / "model.pt",
                      map_location="cpu", weights_only=False)
    # Both SEFusion models share the same class (hidden_dim may differ)
    hidden = 384 if tag == "multimodal_taskB_v2" else 256
    model = SEFusionMultimodalHead(n_out=n_out,
                                   vision_dim=features.CLIP_FEAT_DIM,
                                   hidden_dim=hidden).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(va_txt, va_img).cpu().numpy()
    return expit(logits)


def evaluate_task_b_v3():
    device = features.get_device()

    # ── Data ────────────────────────────────────────────────────────────────
    s_va = data.load_split("val", "B", "multimodal", samples=False, balance=False)
    y_va = s_va.y  # shape [1500, 4]

    # Plain text features
    va_txt_plain = torch.from_numpy(
        features.text_features("val", s_va.ids, s_va.texts, device)
    ).float().to(device)

    # Scene-augmented text features
    scene_va = np.load(config.CACHE_DIR / "xlm_roberta_scene_val.npz")
    va_txt_scene = torch.from_numpy(scene_va["feats"][s_va.ids]).float().to(device)
    # Blended (matches train_task_b_v2.py)
    va_txt_v2 = 0.6 * va_txt_scene + 0.4 * va_txt_plain

    # Image features
    va_img = torch.from_numpy(
        features.clip_features("val", s_va.ids, s_va.image_paths, device)
    ).float().to(device)

    # ── Load Model Probabilities ─────────────────────────────────────────────
    print("Loading model probabilities...")

    # Model A: SEFusion V1 (plain XLM-R)
    probs_A = _load_probs_se("multimodal_taskB_sefusion", 4, device, va_txt_plain, va_img)

    # Model B: SEFusion V2 (scene-blended XLM-R, Focal+LabelSmooth)
    probs_B = _load_probs_se("multimodal_taskB_v2", 4, device, va_txt_v2, va_img)

    # Model C: Text-only head
    ckpt_txt = torch.load(config.OUTPUT_DIR / "text_taskB" / "model.pt",
                          map_location="cpu", weights_only=False)
    head_txt = load_head(ckpt_txt, "text", "B").to(device).eval()
    with torch.no_grad():
        probs_C = expit(head_txt(va_txt_plain).cpu().numpy())

    print(f"  A (SEFusion V1):  shape={probs_A.shape}")
    print(f"  B (SEFusion V2):  shape={probs_B.shape}")
    print(f"  C (Text-only):    shape={probs_C.shape}")

    # ── Grid Search Best Blend Weights ───────────────────────────────────────
    print("\nGrid-searching blend weights (alpha_A, alpha_B, alpha_C)...")
    # Coarse grid: steps of 0.1, summing to 1.0
    steps = np.arange(0, 1.01, 0.1)
    best_avg_macro = -1.0
    best_weights   = (0.4, 0.4, 0.2)

    for a, b in product(steps, steps):
        c = round(1.0 - a - b, 6)
        if c < 0 or c > 1.0001:
            continue
        blend = a * probs_A + b * probs_B + c * probs_C
        # Quick 0.5-threshold macro F1 sum (fast proxy)
        macro_sum = sum(
            f1_score(y_va[:, j], (blend[:, j] > 0.5).astype(int), average="macro")
            for j in range(4)
        )
        if macro_sum > best_avg_macro:
            best_avg_macro = macro_sum
            best_weights   = (float(a), float(b), float(c))

    aW, bW, cW = best_weights
    print(f"  Best weights → A={aW:.1f} B={bW:.1f} C={cW:.1f} "
          f"(raw avg macro={best_avg_macro/4:.4f})")

    # ── Fine-tuned Per-Emotion Threshold Search ──────────────────────────────
    probs_blend = aW * probs_A + bW * probs_B + cW * probs_C

    print("\n" + "=" * 65)
    print("TASK B V3 ENSEMBLE — PER-EMOTION THRESHOLD OPTIMIZATION")
    print(f"Blend: {aW:.0%} SEFusion-V1 + {bW:.0%} SEFusion-V2 + {cW:.0%} Text")
    print("=" * 65)

    thresholds = {}
    preds      = []
    for j, name in enumerate(config.TASK_B_LABELS):
        best_th, best_f1 = 0.5, 0.0
        for th in np.linspace(0.05, 0.95, 181):   # 0.5% step
            mf1 = f1_score(y_va[:, j], (probs_blend[:, j] > th).astype(int),
                           average="macro")
            if mf1 > best_f1:
                best_f1 = mf1
                best_th = th
        thresholds[name] = float(best_th)
        pred_col = (probs_blend[:, j] > best_th).astype(int)
        preds.append(pred_col)

        wf1 = f1_score(y_va[:, j], pred_col, average="weighted")
        acc = accuracy_score(y_va[:, j], pred_col)
        print(f"\n--- {name.upper()} (th={best_th:.3f}) ---")
        print(classification_report(y_va[:, j], pred_col,
                                    target_names=[f"not_{name}", name], digits=4))

    pred_mat     = np.column_stack(preds)
    avg_macro    = float(np.mean([f1_score(y_va[:, j], pred_mat[:, j], average="macro")    for j in range(4)]))
    avg_weighted = float(np.mean([f1_score(y_va[:, j], pred_mat[:, j], average="weighted") for j in range(4)]))
    avg_acc      = float(np.mean([accuracy_score(y_va[:, j], pred_mat[:, j])               for j in range(4)]))

    print("\n" + "=" * 65)
    print("TASK B V3 FINAL ENSEMBLE SUMMARY")
    print("=" * 65)
    print(f"  Average Accuracy   : {avg_acc*100:.2f}%")
    print(f"  Average Weighted F1: {avg_weighted*100:.2f}%")
    print(f"  Average Macro F1   : {avg_macro*100:.2f}%")
    print(f"  Optimal Thresholds : {thresholds}")
    print(f"  Blend Weights      : A={aW:.1f} B={bW:.1f} C={cW:.1f}")

    # ── Comparison with previous best ───────────────────────────────────────
    prev_macro    = 52.60
    prev_weighted = 81.73
    print("\n── vs Previous Best (80% V1 + 20% Text) ──")
    print(f"  Macro F1   : {avg_macro*100:.2f}%  ({avg_macro*100-prev_macro:+.2f}%)")
    print(f"  Weighted F1: {avg_weighted*100:.2f}%  ({avg_weighted*100-prev_weighted:+.2f}%)")


if __name__ == "__main__":
    evaluate_task_b_v3()
