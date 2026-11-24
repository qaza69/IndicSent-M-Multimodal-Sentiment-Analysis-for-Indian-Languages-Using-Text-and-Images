"""Download openai/clip-vit-large-patch14 and extract features for Memotion 3.

Processes:
- train: 7,000 images -> cache/clip_large_train.npz (7000 x 768)
- val:   1,500 images -> cache/clip_large_val.npz   (1500 x 768)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection

import config


class MemeImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt / missing images
            print(f"[Warning] Failed to load {path}: {e}")
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (128, 128, 128))
        return img


class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        inputs = self.processor(images=batch, return_tensors="pt")
        return inputs["pixel_values"]


def extract_split(split_name, img_dir, csv_path, processor, model, device, batch_size=32):
    out_file = config.CACHE_DIR / f"clip_large_{split_name}.npz"
    if out_file.exists():
        print(f"[{split_name}] Already cached at {out_file} - skipping extraction.")
        return

    df = pd.read_csv(csv_path)
    ids = df["image_id"].astype(int).tolist()
    img_paths = [str(img_dir / f"{i}.jpg") for i in ids]

    dataset = MemeImageDataset(img_paths)
    collate = CollateFn(processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    all_feats = []
    print(f"\n[{split_name}] Extracting features for {len(img_paths)} images using CLIP ViT-L/14...")
    t0 = time.time()

    with torch.no_grad():
        for batch_pixel_values in tqdm(loader, desc=f"CLIP ViT-L/14 [{split_name}]"):
            pixel_values = batch_pixel_values.to(device)
            out = model(pixel_values=pixel_values)
            # image_embeds is [B, 768] projected and normalized
            embeds = out.image_embeds.cpu().numpy()
            # L2 normalize
            norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeds = embeds / norms
            all_feats.append(embeds.astype(np.float32))

    feats = np.concatenate(all_feats, axis=0)
    print(f"Extracted shape: {feats.shape} in {time.time() - t0:.1f}s")

    np.savez_compressed(out_file, ids=np.array(ids), feats=feats)
    print(f"Saved -> {out_file}")


def main():
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"Extraction Device: {device} (Apple M5)")

    # Find local snapshot
    local_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14" / "snapshots" / "32bd64288804d66eefd0ccbe215aa642df71cc41"
    model_name = str(local_path) if local_path.exists() else "openai/clip-vit-large-patch14"
    print(f"Loading from: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device).eval()

    # Val first (fast sanity check)
    extract_split("val", config.VAL_IMG_DIR, config.VAL_CSV, processor, model, device, batch_size=32)

    # Train (7k images)
    extract_split("train", config.TRAIN_IMG_DIR, config.TRAIN_CSV, processor, model, device, batch_size=32)

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    main()
