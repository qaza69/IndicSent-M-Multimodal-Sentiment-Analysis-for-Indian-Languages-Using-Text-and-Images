"""
Trainable LoRA (Low-Rank Adaptation) fine-tuning for Hinglish Memes.
Adapts XLM-RoBERTa transformer layers specifically for Hinglish code-switched meme slang and sentiment.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

import config
import data
import features
from models import build_head, CLIPMultimodalHead, TextHead


class MemeDataset(Dataset):
    def __init__(self, texts, labels, clip_feats=None):
        self.texts = texts
        self.labels = labels
        self.clip_feats = clip_feats

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }
        if self.clip_feats is not None:
            item["clip_feat"] = self.clip_feats[idx]
        return item


def collate_fn(batch, tokenizer, max_len=64):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    res = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }
    if "clip_feat" in batch[0]:
        clip_feats = torch.stack([b["clip_feat"] for b in batch])
        res["clip_feat"] = clip_feats
    return res


def parse_args():
    p = argparse.ArgumentParser(description="LoRA Fine-tuning for Hinglish Memes")
    p.add_argument("--model", choices=["text", "multimodal"], default="text")
    p.add_argument("--task", choices=["A", "B"], default="A")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-len", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = features.get_device()
    print(f"=== LoRA Fine-Tuning | model={args.model} | task={args.task} | device={device} ===")

    # 1. Load data
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)

    if args.task == "A":
        y_train = train_df["overall"].map(data.TASK_A_MAP).to_numpy()
        y_val = val_df["overall"].map(data.TASK_A_MAP).to_numpy()
        n_out = 3
    else:
        y_train = train_df[["humour", "sarcastic", "offensive", "motivational"]].copy()
        # For multi-label binary
        for c in config.TASK_B_LABELS:
            y_train[c] = y_train[c].map(data.TASK_B_MAPS[c])
        y_train = y_train.to_numpy(dtype=np.float32)
        n_out = 4

    train_texts = train_df["ocr"].fillna(".").astype(str).str.replace("\n", " ", regex=False).tolist()
    val_texts = val_df["ocr"].fillna(".").astype(str).str.replace("\n", " ", regex=False).tolist()

    train_clip, val_clip = None, None
    if args.model == "multimodal":
        tr_ids = train_df["image_id"].astype(int).tolist()
        va_ids = val_df["image_id"].astype(int).tolist()
        tr_imgs = [str(config.TRAIN_IMG_DIR / f"{i}.jpg") for i in tr_ids]
        va_imgs = [str(config.VAL_IMG_DIR / f"{i}.jpg") for i in va_ids]
        print("Loading cached CLIP visual embeddings...")
        train_clip = torch.from_numpy(features.clip_features("train", tr_ids, tr_imgs, device)).float()
        val_clip = torch.from_numpy(features.clip_features("val", va_ids, va_imgs, device)).float()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    train_ds = MemeDataset(train_texts, y_train, train_clip)
    val_ds = MemeDataset(val_texts, y_val, val_clip)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len)
    )

    # 2. Base encoder + LoRA config
    print(f"Initializing {config.TEXT_MODEL_NAME} with LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    base_text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    text_encoder = get_peft_model(base_text_model, lora_config).to(device)
    text_encoder.print_trainable_parameters()

    # 3. Head
    head = build_head(args.model, args.task, vision_backbone="clip").to(device)

    # 4. Optimizer & Loss
    params = list(text_encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.task == "A":
        classes, counts = np.unique(y_train, return_counts=True)
        weights = torch.tensor(len(y_train) / (len(classes) * (counts ** 0.8)), dtype=torch.float32).to(device)
        weights = weights / weights.mean()
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_metrics = None
    best_epoch = 1

    tag = f"{args.model}_task{args.task}_lora"
    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        text_encoder.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            txt_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

            if args.model == "multimodal":
                clip_feat = batch["clip_feat"].to(device)
                logits = head(txt_out, clip_feat)
            else:
                logits = head(txt_out)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch {ep}/{args.epochs} [{step+1}/{len(train_loader)}] Loss: {total_loss/total:.4f} Acc: {correct/total:.4f}")

        scheduler.step()

        # Validation
        text_encoder.eval()
        head.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                txt_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                if args.model == "multimodal":
                    clip_feat = batch["clip_feat"].to(device)
                    logits = head(txt_out, clip_feat)
                else:
                    logits = head(txt_out)

                loss = loss_fn(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

        dt = time.time() - t0
        print(f"==> Epoch {ep} ({dt:.1f}s) | Val Loss: {val_loss/len(all_labels):.4f} | Val Acc: {acc:.4f} | Weighted F1: {f1_w:.4f} | Macro F1: {f1_m:.4f}")

        if f1_w > best_val_f1:
            best_val_f1 = f1_w
            best_epoch = ep
            best_metrics = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1_w),
                "macro_f1": float(f1_m),
                "epoch": ep
            }
            # Save Head and LoRA Adapter
            torch.save({
                "state_dict": head.state_dict(),
                "model": args.model,
                "task": args.task,
                "has_lora": True,
                "vision_backbone": "clip"
            }, out_dir / "model.pt")
            text_encoder.save_pretrained(out_dir / "lora_adapter")
            (out_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2))
            print(f"⭐ Saved new best LoRA model checkpoint to {out_dir}!")

    print(f"\n==========================================")
    print(f"LoRA Training Complete!")
    print(f"Best Validation Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best Weighted F1:         {best_metrics['f1']:.4f}")
    print(f"Best Macro F1:            {best_metrics['macro_f1']:.4f}")
    print(f"Saved checkpoint & adapter at: {out_dir}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
