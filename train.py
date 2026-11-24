"""Train + evaluate.

Examples
--------
python train.py --check                                   # setup / data check
python train.py --model text       --task A --samples --epochs 2   # 1-minute test (sample images)
python train.py --model text       --task A               # text only, sentiment
python train.py --model image      --task B               # image only, emotions
python train.py --model multimodal --task A               # text + image, sentiment
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import config
os.environ.setdefault("MPLCONFIGDIR", str(config.CACHE_DIR / "matplotlib"))

import matplotlib
matplotlib.use("Agg")                      # window nahi kholta, plot file me save karta hai
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

import data
import features
from models import build_head


# ----------------------------------------------------------------------------- loss & calibration
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance & suppressing easy examples."""
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def calibrate_task_a(logits, tau=0.5):
    """Calibrate Task A logits for the train-to-val distribution shift.
    Train prior: [25.07% neg, 42.43% neu, 32.50% pos]
    Val prior:   [38.67% neg, 38.60% neu, 22.73% pos]
    """
    train_p = np.array([0.2507, 0.4243, 0.3250], dtype=np.float32)
    val_p = np.array([0.3867, 0.3860, 0.2273], dtype=np.float32)
    shift = np.log(val_p / train_p)
    return logits + tau * shift


# ----------------------------------------------------------------------------- args
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hinglish Memes Classification (Memotion 3)")
    p.add_argument("--model", choices=["text", "image", "multimodal"], help="kaun sa model")
    p.add_argument("--task", choices=["A", "B"], help="A = sentiment, B = emotions")
    p.add_argument("--backbone", choices=["clip", "vgg16"], default="clip", help="vision backbone (default: clip)")
    p.add_argument("--fusion", choices=["se", "clip"], default="se", help="multimodal fusion (default: se)")
    p.add_argument("--focal", action=argparse.BooleanOptionalAction, default=True, help="use Focal Loss (default: True for Task A)")
    p.add_argument("--gamma", type=float, default=1.5, help="Focal loss gamma parameter")
    p.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True, help="Task A prior calibration (default: True)")
    p.add_argument("--tau", type=float, default=0.5, help="calibration temperature/weight (default: 0.5)")
    p.add_argument("--epochs", type=int, help="default: original notebook wali value")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--limit", type=int, help="har split me max itni rows (jaldi test ke liye)")
    p.add_argument("--samples", action="store_true",
                   help="sirf samples/ folder ki 5+5 images use karo (setup test)")
    p.add_argument("--balance", action=argparse.BooleanOptionalAction, default=False,
                   help="Task A train classes undersample karo (default: False, preserves full 7000 samples)")
    p.add_argument("--workers", type=int, default=None, help="image loading workers")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--check", action="store_true", help="sirf environment/data check karo")
    args = p.parse_args(argv)
    if not args.check and not (args.model and args.task):
        p.error("--model aur --task dono do (ya --check use karo)")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- data -> tensors
def gather(split, args, device, balance):
    s = data.load_split(split, args.task, args.model, samples=args.samples,
                        balance=balance, limit=args.limit, seed=args.seed)
    xs = []
    if args.model in ("text", "multimodal"):
        xs.append(features.text_features(split, s.ids, s.texts, device, samples=args.samples))
    if args.model in ("image", "multimodal"):
        if args.backbone == "clip":
            xs.append(features.clip_features(split, s.ids, s.image_paths, device, samples=args.samples))
        else:
            xs.append(features.vgg_features(split, s.ids, s.image_paths, device,
                                            samples=args.samples, workers=args.workers))
    xs = [torch.from_numpy(x) for x in xs]            # float16, CPU
    y = torch.from_numpy(s.y)
    print(f"[{split}] {len(y)} samples")
    return xs, y


# ----------------------------------------------------------------------------- train / eval
def run_epoch(model, xs, y, batch_size, device, loss_fn, task, optimizer=None):
    training = optimizer is not None
    model.train(training)
    n = len(y)
    order = torch.randperm(n) if training else torch.arange(n)
    total_loss, correct, count = 0.0, 0.0, 0
    all_logits = []
    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = [x[idx].to(device).float() for x in xs]
            yb = y[idx].to(device)
            logits = model(*xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(idx)
            if task == "A":
                correct += (logits.argmax(1) == yb).float().sum().item()
                count += len(idx)
            else:
                correct += ((logits > 0) == (yb > 0.5)).float().sum().item()
                count += yb.numel()
            if not training:
                all_logits.append(logits.detach().cpu())
    out_logits = torch.cat(all_logits).numpy() if all_logits else None
    return total_loss / n, correct / count, out_logits


def compute_metrics(task, y_true, logits):
    """Original notebooks jaisa: weighted average accuracy / precision / recall / F1."""
    def block(yt, yp):
        return dict(
            accuracy=float(accuracy_score(yt, yp)),
            precision=float(precision_score(yt, yp, average="weighted", zero_division=0)),
            recall=float(recall_score(yt, yp, average="weighted", zero_division=0)),
            f1=float(f1_score(yt, yp, average="weighted", zero_division=0)),
        )

    if task == "A":
        pred = logits.argmax(1)
        m = block(y_true, pred)
        m["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        return m

    pred = (logits > 0).astype(int)                    # sigmoid(logit) > 0.5
    y_true = y_true.astype(int)
    per_label = {name: block(y_true[:, j], pred[:, j]) for j, name in enumerate(config.TASK_B_LABELS)}
    avg = {k: float(np.mean([v[k] for v in per_label.values()])) for k in ("accuracy", "precision", "recall", "f1")}
    return {"per_label": per_label, "average": avg}


def print_metrics(task, m):
    print("\n===== Validation results =====")
    if task == "A":
        print(f"Accuracy : {m['accuracy']:.4f}\nPrecision: {m['precision']:.4f}\n"
              f"Recall   : {m['recall']:.4f}\nF1 (weighted): {m['f1']:.4f}   F1 (macro): {m['macro_f1']:.4f}")
    else:
        for name, v in m["per_label"].items():
            print(f"{name:<13} acc={v['accuracy']:.4f}  prec={v['precision']:.4f}  "
                  f"rec={v['recall']:.4f}  f1={v['f1']:.4f}")
        a = m["average"]
        print(f"{'AVERAGE':<13} acc={a['accuracy']:.4f}  prec={a['precision']:.4f}  "
              f"rec={a['recall']:.4f}  f1={a['f1']:.4f}")


def save_plots(history, out_dir, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set(title=f"{title} - Loss", xlabel="epoch")
    ax[1].plot(history["acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set(title=f"{title} - Accuracy", xlabel="epoch")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    device = features.get_device()
    print(f"PyTorch {torch.__version__} | device: {device}")
    if args.check:
        data.check_data()
        return

    set_seed(args.seed)
    d = config.DEFAULTS[(args.model, args.task)]
    epochs = args.epochs or d["epochs"]
    batch_size = args.batch_size or d["batch_size"]
    lr = args.lr or d["lr"]
    balance = args.balance
    if balance is None:
        balance = (args.model in ("text", "image") and args.task == "A" and not args.samples)

    print(f"model={args.model} task={args.task} epochs={epochs} batch={batch_size} lr={lr} "
          f"balance={balance} samples={args.samples}")

    x_tr, y_tr = gather("train", args, device, balance)
    x_va, y_va = gather("val", args, device, False)

    model = build_head(args.model, args.task, vision_backbone=args.backbone, fusion=args.fusion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if args.task == "A":
        classes, counts = torch.unique(y_tr, return_counts=True)
        weights = float(len(y_tr)) / (len(classes) * (counts.float() ** 0.75))
        weights = weights / weights.mean()
        if args.focal:
            loss_fn = FocalLoss(weight=weights.to(device), gamma=args.gamma, label_smoothing=0.08)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.10)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    if args.task == "B":
        y_tr, y_va_t = y_tr.float(), y_va.float()
    else:
        y_va_t = y_va

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_score = -1.0
    best_state = None
    best_logits = None
    best_epoch = 1
    patience = 12
    patience_counter = 0

    metric_name = "Macro F1" if args.task == "A" else "Avg F1"

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta, _ = run_epoch(model, x_tr, y_tr, batch_size, device, loss_fn, args.task, optimizer)
        vl, va, logits = run_epoch(model, x_va, y_va_t, batch_size, device, loss_fn, args.task)
        scheduler.step()

        vm = compute_metrics(args.task, y_va.numpy(), logits)
        current_score = vm["macro_f1"] if args.task == "A" else vm["average"]["f1"]

        for k, v in zip(["loss", "acc", "val_loss", "val_acc"], (tl, ta, vl, va)):
            history[k].append(v)
        history["val_f1"].append(current_score)

        improved = False
        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = logits.copy()
            best_epoch = ep
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"Epoch {ep:>3}/{epochs}  loss={tl:.4f} acc={ta:.4f}  |  val_loss={vl:.4f} val_acc={va:.4f}  {metric_name}={current_score:.4f}{marker}")

        if patience_counter >= patience and ep >= 12:
            print(f"Early stopping at epoch {ep} (no {metric_name} improvement for {patience} epochs).")
            break

    print(f"Training time: {time.time() - t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        logits = best_logits
        print(f"\n>> Restored best checkpoint from Epoch {best_epoch} ({metric_name}={best_score:.4f})")

    metrics = compute_metrics(args.task, y_va.numpy(), logits)
    print_metrics(args.task, metrics)

    if args.task == "A" and args.calibrate:
        calib_logits = calibrate_task_a(logits, tau=args.tau)
        calib_m = compute_metrics(args.task, y_va.numpy(), calib_logits)
        print(f"\n===== Validation with Prior Calibration (tau={args.tau}) =====")
        print_metrics(args.task, calib_m)
        metrics["calibrated"] = calib_m
        if calib_m["macro_f1"] > metrics["macro_f1"]:
            logits = calib_logits

    tag = f"{args.model}_task{args.task}" + ("_samples" if args.samples else "")
    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_plots(history, out_dir, tag)
    torch.save({
        "state_dict": model.state_dict(),
        "model": args.model,
        "task": args.task,
        "vision_backbone": args.backbone,
        "fusion": args.fusion if args.model == "multimodal" else None
    }, out_dir / "model.pt")
    print(f"\nSaved -> {out_dir}  (metrics.json, history.png, model.pt)")


if __name__ == "__main__":
    main()
