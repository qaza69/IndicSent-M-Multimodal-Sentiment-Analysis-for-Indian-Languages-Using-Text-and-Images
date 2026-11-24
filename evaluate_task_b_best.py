"""Task B BEST Ensemble Evaluator — Final Production Version.

Best configuration found via exhaustive grid search:
  50% SEFusion V1 (plain XLM-R + CLIP, BCE+pos_weight)
+ 50% SEFusion V2 (scene XLM-R blend + CLIP, Focal+LabelSmooth)

Per-emotion thresholds optimized on validation set.

Benchmarks (validation):
  Baseline (old)      : WF1=76.93%  MacroF1=45.60%
  80% V1 + 20% Text   : WF1=81.73%  MacroF1=52.60%  (previous best)
  50% V1 + 50% V2     : WF1=81.47%  MacroF1=53.26%  (THIS — +0.66% macro)
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.special import expit

import config
import data
import features
from models import SEFusionMultimodalHead


def evaluate_task_b_best():
    device = features.get_device()

    s_va = data.load_split("val", "B", "multimodal", samples=False, balance=False)
    y_va = s_va.y  # [1500, 4]

    # Plain text features
    va_txt_plain = torch.from_numpy(
        features.text_features("val", s_va.ids, s_va.texts, device)
    ).float().to(device)

    # Scene-augmented text features (matches V2 training)
    sc_va = np.load(config.CACHE_DIR / "xlm_roberta_scene_val.npz")
    va_txt_v2 = (0.6 * torch.from_numpy(sc_va["feats"][s_va.ids]).float()
                 + 0.4 * va_txt_plain.cpu()).to(device)

    # Image features
    va_img = torch.from_numpy(
        features.clip_features("val", s_va.ids, s_va.image_paths, device)
    ).float().to(device)

    def _load_se(tag, hidden, va_txt):
        ckpt  = torch.load(config.OUTPUT_DIR / tag / "model.pt",
                           map_location="cpu", weights_only=False)
        model = SEFusionMultimodalHead(
            n_out=4, vision_dim=features.CLIP_FEAT_DIM, hidden_dim=hidden
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            return expit(model(va_txt, va_img).cpu().numpy())

    print("Loading models...")
    probs_V1  = _load_se("multimodal_taskB_sefusion", 256, va_txt_plain)
    probs_V2  = _load_se("multimodal_taskB_v2",       384, va_txt_v2)
    blend     = 0.5 * probs_V1 + 0.5 * probs_V2

    print("\n" + "=" * 65)
    print("TASK B BEST ENSEMBLE — 50% SEFusion-V1 + 50% SEFusion-V2")
    print("=" * 65)

    preds = []
    for j, name in enumerate(config.TASK_B_LABELS):
        best_th, best_f1 = 0.5, 0.0
        for th in np.linspace(0.05, 0.95, 181):
            mf1 = f1_score(y_va[:, j], (blend[:, j] > th).astype(int), average="macro")
            if mf1 > best_f1:
                best_f1 = mf1; best_th = th
        pred_col = (blend[:, j] > best_th).astype(int)
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

    print("=" * 65)
    print("TASK B BEST ENSEMBLE — FINAL SUMMARY")
    print("=" * 65)
    print(f"  Avg Accuracy   : {avg_acc*100:.2f}%")
    print(f"  Avg Weighted F1: {avg_weighted*100:.2f}%")
    print(f"  Avg Macro F1   : {avg_macro*100:.2f}%")
    print(f"\n  vs Baseline    : +{(avg_macro-0.456)*100:.2f}% Macro, +{(avg_weighted-0.7693)*100:.2f}% Weighted")
    print(f"  vs Prev Best   : +{(avg_macro-0.526)*100:.2f}% Macro, +{(avg_weighted-0.8173)*100:.2f}% Weighted")


if __name__ == "__main__":
    evaluate_task_b_best()
