"""Task B Mega-Ensemble — Best possible blend.

Per emotion, we have 3 sources:
  V1: multimodal_taskB_sefusion   (plain XLM-R + CLIP, BCE+pos_weight)
  V2: multimodal_taskB_v2         (scene XLM-R blend + CLIP, Focal+LabelSmooth)
  PE: taskB_per_emotion/<emotion> (dedicated binary classifier)

We grid-search per-emotion blend weights (alpha_V1, alpha_V2, alpha_PE)
independently for each emotion, then apply optimal thresholds.
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.special import expit
from itertools import product

import config
import data
import features
from models import SEFusionMultimodalHead


def _get_probs_sefusion(tag, hidden, va_txt, va_img, device):
    ckpt  = torch.load(config.OUTPUT_DIR / tag / "model.pt",
                       map_location="cpu", weights_only=False)
    model = SEFusionMultimodalHead(n_out=4, vision_dim=features.CLIP_FEAT_DIM,
                                    hidden_dim=hidden).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        return expit(model(va_txt, va_img).cpu().numpy())   # [1500, 4]


def _get_probs_per_emotion(emotion_name, va_txt, va_img, device):
    ckpt  = torch.load(config.OUTPUT_DIR / "taskB_per_emotion" / emotion_name / "model.pt",
                       map_location="cpu", weights_only=False)
    model = SEFusionMultimodalHead(n_out=1, vision_dim=features.CLIP_FEAT_DIM,
                                    hidden_dim=256).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        return expit(model(va_txt, va_img).cpu().numpy().squeeze())   # [1500]


def evaluate_mega_ensemble():
    device = features.get_device()

    s_va    = data.load_split("val", "B", "multimodal", samples=False, balance=False)
    y_va    = s_va.y   # [1500, 4]

    # Plain text features
    va_txt_plain = torch.from_numpy(
        features.text_features("val", s_va.ids, s_va.texts, device)
    ).float().to(device)

    # Scene-augmented text features (matches train_task_b_v2.py)
    sc_va    = np.load(config.CACHE_DIR / "xlm_roberta_scene_val.npz")
    va_txt_v2 = (0.6 * torch.from_numpy(sc_va["feats"][s_va.ids]).float()
                 + 0.4 * va_txt_plain.cpu()).to(device)

    # Image features
    va_img = torch.from_numpy(
        features.clip_features("val", s_va.ids, s_va.image_paths, device)
    ).float().to(device)

    # ── Load All Probabilities ───────────────────────────────────────────────
    print("Loading model probabilities...")
    probs_V1 = _get_probs_sefusion("multimodal_taskB_sefusion", 256,
                                    va_txt_plain, va_img, device)   # [1500, 4]
    probs_V2 = _get_probs_sefusion("multimodal_taskB_v2", 384,
                                    va_txt_v2, va_img, device)       # [1500, 4]

    probs_PE = {}   # per-emotion, each shape [1500]
    for name in config.TASK_B_LABELS:
        probs_PE[name] = _get_probs_per_emotion(name, va_txt_v2, va_img, device)
    print("  All models loaded.")

    # ── Per-Emotion Weight Search ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MEGA-ENSEMBLE — Per-Emotion Blend Weight Search")
    print("=" * 65)

    steps   = np.arange(0, 1.01, 0.1)
    final_preds = []
    final_info  = {}

    for j, name in enumerate(config.TASK_B_LABELS):
        y_col      = y_va[:, j]
        v1_col     = probs_V1[:, j]
        v2_col     = probs_V2[:, j]
        pe_col     = probs_PE[name]

        best_mf1   = -1.0
        best_w     = (0.5, 0.5, 0.0)

        for a, b in product(steps, steps):
            c = round(1.0 - a - b, 6)
            if c < 0 or c > 1.0001:
                continue
            blend_col = a * v1_col + b * v2_col + c * pe_col
            # Quick macro-F1 at 0.5 (fast proxy for ranking)
            mf1 = f1_score(y_col, (blend_col > 0.5).astype(int), average="macro")
            if mf1 > best_mf1:
                best_mf1 = mf1
                best_w   = (float(a), float(b), float(c))

        aW, bW, cW = best_w
        blend_col  = aW * v1_col + bW * v2_col + cW * pe_col

        # Fine threshold search
        best_th, best_f1 = 0.5, 0.0
        for th in np.linspace(0.05, 0.95, 181):
            mf1 = f1_score(y_col, (blend_col > th).astype(int), average="macro")
            if mf1 > best_f1:
                best_f1 = mf1; best_th = th

        pred_col = (blend_col > best_th).astype(int)
        wf1  = f1_score(y_col, pred_col, average="weighted")
        acc  = accuracy_score(y_col, pred_col)

        final_preds.append(pred_col)
        final_info[name] = {
            "weights"  : best_w,
            "threshold": best_th,
            "macro_f1" : best_f1,
            "weighted_f1": wf1,
            "accuracy" : acc,
        }
        print(f"\n  {name.upper()}: V1={aW:.1f} V2={bW:.1f} PE={cW:.1f} | "
              f"th={best_th:.3f} | WF1={wf1:.4f} | MacroF1={best_f1:.4f} | Acc={acc:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    pred_mat     = np.column_stack(final_preds)
    avg_macro    = float(np.mean([f1_score(y_va[:, j], pred_mat[:, j], average="macro")    for j in range(4)]))
    avg_weighted = float(np.mean([f1_score(y_va[:, j], pred_mat[:, j], average="weighted") for j in range(4)]))
    avg_acc      = float(np.mean([accuracy_score(y_va[:, j], pred_mat[:, j])               for j in range(4)]))

    print("\n" + "=" * 65)
    print("TASK B MEGA-ENSEMBLE FINAL RESULTS")
    print("=" * 65)
    print(f"  Avg Accuracy   : {avg_acc*100:.2f}%")
    print(f"  Avg Weighted F1: {avg_weighted*100:.2f}%")
    print(f"  Avg Macro F1   : {avg_macro*100:.2f}%")

    print("\n── vs Previous Bests ──")
    print(f"  Baseline (old)        : WF1=76.93%  MacroF1=45.60%")
    print(f"  SEFusion V1           : WF1=81.45%  MacroF1=52.46%")
    print(f"  80% V1 + 20% Text     : WF1=81.73%  MacroF1=52.60%")
    print(f"  50% V1 + 50% V2       : WF1=81.47%  MacroF1=53.26%")
    print(f"  MEGA-ENSEMBLE         : WF1={avg_weighted*100:.2f}%  MacroF1={avg_macro*100:.2f}%  ← NEW")

    print("\n── Per-Emotion Detail ──")
    for name, info in final_info.items():
        w = info['weights']
        print(f"  {name:<14} | V1={w[0]:.1f} V2={w[1]:.1f} PE={w[2]:.1f} | "
              f"th={info['threshold']:.3f} | WF1={info['weighted_f1']*100:.2f}% | "
              f"MacroF1={info['macro_f1']*100:.2f}%")


if __name__ == "__main__":
    evaluate_mega_ensemble()
