"""Level 2: End-to-End Multi-Task LoRA Adaptation for Memotion 3.

Adapts top transformer layers of XLM-RoBERTa (layers 8-11) specifically
for Hinglish code-mixed meme slang and humor, coupled with the
MultiTaskSEFusionHead initialized from our best Level 1 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

import config
import data
import features
from models import MultiTaskSEFusionHead, load_head
from train import FocalLoss, calibrate_task_a, compute_metrics


class MultimodalMemeDataset(Dataset):
    def __init__(self, texts, clip_feats, labels_a, labels_b):
        self.texts = texts
        self.clip_feats = clip_feats
        self.labels_a = labels_a
        self.labels_b = labels_b

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "clip_feat": self.clip_feats[idx],
            "label_a": self.labels_a[idx],
            "label_b": self.labels_b[idx],
        }


def make_collate_fn(tokenizer, max_len=64):
    def collate(batch):
        texts = [b["text"] for b in batch]
        clip_feats = torch.stack([b["clip_feat"] for b in batch])
        labels_a = torch.tensor([b["label_a"] for b in batch], dtype=torch.long)
        labels_b = torch.tensor(np.array([b["label_b"] for b in batch]), dtype=torch.float32)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "clip_feat": clip_feats,
            "label_a": labels_a,
            "label_b": labels_b,
        }
    return collate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lora-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--alpha-b", type=float, default=0.15)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)

    device = features.get_device()
    print("=" * 65)
    print(f"LEVEL 2: Multi-Task LoRA Adaptation (Device: {device})")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LoRA LR: {args.lora_lr} | Head LR: {args.head_lr}")
    print(f"LoRA Rank: {args.lora_r} | Alpha: {args.lora_alpha} | Emotion Alpha: {args.alpha_b}")
    print("=" * 65)

    # 1. Load Data Splits
    s_tr_a = data.load_split("train", "A", "multimodal", samples=False, balance=False)
    s_tr_b = data.load_split("train", "B", "multimodal", samples=False, balance=False)
    s_va_a = data.load_split("val", "A", "multimodal", samples=False, balance=False)
    s_va_b = data.load_split("val", "B", "multimodal", samples=False, balance=False)

    print("Loading cached visual CLIP embeddings...")
    tr_clip = torch.from_numpy(features.clip_features("train", s_tr_a.ids, s_tr_a.image_paths, device)).float()
    va_clip = torch.from_numpy(features.clip_features("val", s_va_a.ids, s_va_a.image_paths, device)).float()

    # Clean texts
    tr_texts = [str(t).replace("\n", " ").strip() if str(t).strip() else "." for t in s_tr_a.texts]
    va_texts = [str(t).replace("\n", " ").strip() if str(t).strip() else "." for t in s_va_a.texts]

    tok, base_text_model = features.load_text_encoder()

    train_ds = MultimodalMemeDataset(tr_texts, tr_clip, s_tr_a.y, s_tr_b.y)
    val_ds = MultimodalMemeDataset(va_texts, va_clip, s_va_a.y, s_va_b.y)

    collate = make_collate_fn(tok, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 2. Setup LoRA on top 4 layers (8, 9, 10, 11)
    print(f"Applying LoRA on top layers (8-11) of {config.TEXT_MODEL_NAME}...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        layers_to_transform=[8, 9, 10, 11],
        lora_dropout=0.1,
        bias="none",
    )
    text_model = get_peft_model(base_text_model, lora_config).to(device)
    text_model.print_trainable_parameters()

    # 3. Setup Head (Warm-started from best Level 1 MultiTask checkpoint)
    head = MultiTaskSEFusionHead().to(device)
    init_ckpt_path = config.OUTPUT_DIR / "multitask_multimodal" / "model.pt"
    if init_ckpt_path.exists():
        print(f"Warm-starting Head from best Level 1 checkpoint: {init_ckpt_path}")
        init_ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        head.load_state_dict(init_ckpt["state_dict"])
    else:
        print("Initializing head from scratch...")

    # 4. Losses & Parameter Groups
    classes, counts = torch.unique(torch.tensor(s_tr_a.y), return_counts=True)
    weights = float(len(s_tr_a.y)) / (len(classes) * (counts.float() ** 0.75))
    weights = (weights / weights.mean()).to(device)
    loss_fn_a = FocalLoss(weight=weights, gamma=1.5, label_smoothing=0.08)
    loss_fn_b = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW([
        {"params": [p for p in text_model.parameters() if p.requires_grad], "lr": args.lora_lr, "weight_decay": 0.01},
        {"params": head.parameters(), "lr": args.head_lr, "weight_decay": 0.01},
    ])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_score = -1.0
    best_tau_score = -1.0
    best_tau = 0.0
    best_state_head = None
    best_epoch = 1

    out_dir = config.OUTPUT_DIR / "multitask_lora_backbone"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting End-to-End LoRA Training ({args.epochs} epochs)...")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        text_model.train()
        head.train()
        tl, correct, n_samples = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clip_feat = batch["clip_feat"].to(device)
            ya = batch["label_a"].to(device)
            yb = batch["label_b"].to(device)

            # Forward text backbone
            txt_out = text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            la_out, lb_out = head(txt_out, clip_feat)

            la = loss_fn_a(la_out, ya)
            lb = loss_fn_b(lb_out, yb)
            loss = la + args.alpha_b * lb

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bs = len(ya)
            tl += loss.item() * bs
            correct += (la_out.argmax(1) == ya).sum().item()
            n_samples += bs

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch {ep}/{args.epochs} [{step+1:>3}/{len(train_loader)}] Loss: {tl/n_samples:.4f} Acc: {correct/n_samples:.4f}")

        # Evaluation on Validation Split
        text_model.eval()
        head.eval()
        val_la_list, val_ya_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                clip_feat = batch["clip_feat"].to(device)
                txt_out = text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                la_out, _ = head(txt_out, clip_feat)
                val_la_list.append(la_out.cpu().numpy())
                val_ya_list.append(batch["label_a"].numpy())

        v_logits = np.concatenate(val_la_list, axis=0)
        v_y = np.concatenate(val_ya_list, axis=0)
        m = compute_metrics("A", v_y, v_logits)
        raw_mf1 = m["macro_f1"]

        # Fast calibration sweep
        ep_best_tau_score, ep_best_tau = raw_mf1, 0.0
        for tau in np.linspace(0.02, 0.25, 24):
            calib = calibrate_task_a(v_logits, tau=tau)
            calib_mf1 = f1_score(v_y, calib.argmax(1), average="macro")
            if calib_mf1 > ep_best_tau_score:
                ep_best_tau_score = calib_mf1
                ep_best_tau = tau

        improved = ep_best_tau_score > best_tau_score
        if improved:
            best_score = raw_mf1
            best_tau_score = ep_best_tau_score
            best_tau = ep_best_tau
            best_epoch = ep
            best_state_head = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            # Save checkpoint & adapter
            torch.save({
                "state_dict": best_state_head,
                "model": "multitask_lora",
                "vision_backbone": "clip",
                "best_tau": best_tau,
                "raw_macro_f1": best_score,
                "calibrated_macro_f1": best_tau_score,
                "epoch": ep
            }, out_dir / "model.pt")
            text_model.save_pretrained(out_dir / "lora_adapter")
            np.savez_compressed(out_dir / "val_predictions.npz", logits=v_logits, y=v_y)

        dt = time.time() - t0
        print(f"==> Epoch {ep} ({dt:.1f}s) | Val Raw Macro F1: {raw_mf1:.4f} | Calibrated: {ep_best_tau_score:.4f} (tau={ep_best_tau:.3f}) {'🏆 NEW BEST' if improved else ''}")

    print("\n" + "=" * 65)
    print(f"LEVEL 2 TRAINING COMPLETE (Best Epoch: {best_epoch})")
    print(f"Best Calibrated Macro F1: {best_tau_score:.4f} (tau={best_tau:.3f})")
    print(f"Artifacts saved in: {out_dir}")
    print("=" * 65)
    return best_tau_score


if __name__ == "__main__":
    train()
