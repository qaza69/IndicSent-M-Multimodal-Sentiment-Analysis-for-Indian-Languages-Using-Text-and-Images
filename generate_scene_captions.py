"""Batch generate BLIP visual scene captions for Memotion 3 memes.

1. Generates captions:
   - cache/blip_captions_val.json   (1,500 memes)
   - cache/blip_captions_train.json (7,000 memes)
2. Augments OCR text:
   augmented_text = f"{ocr} | Visual scene: {caption}"
3. Extracts XLM-RoBERTa features for the augmented text:
   - cache/xlm_roberta_augmented_val.npz
   - cache/xlm_roberta_augmented_train.npz
"""
import json
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

import config
import features


class ImageBatchDataset(Dataset):
    def __init__(self, ids, img_paths):
        self.ids = ids
        self.img_paths = img_paths

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.ids[idx], img


class CollateImages:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        ids = [b[0] for b in batch]
        imgs = [b[1] for b in batch]
        inputs = self.processor(images=imgs, return_tensors="pt")
        return ids, inputs["pixel_values"]


def generate_split_captions(split_name, img_dir, csv_path, processor, model, device, batch_size=32):
    cache_path = config.CACHE_DIR / f"blip_captions_{split_name}.json"
    if cache_path.exists():
        print(f"[{split_name}] Captions already cached at {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    df = pd.read_csv(csv_path)
    ids = df["image_id"].astype(int).tolist()
    img_paths = [str(img_dir / f"{i}.jpg") for i in ids]

    dataset = ImageBatchDataset(ids, img_paths)
    collate = CollateImages(processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    captions_dict = {}
    print(f"\n[{split_name}] Generating BLIP captions for {len(ids)} memes...")
    t0 = time.time()

    with torch.no_grad():
        for batch_ids, pixel_values in tqdm(loader, desc=f"BLIP [{split_name}]"):
            pixel_values = pixel_values.to(device)
            out = model.generate(pixel_values=pixel_values, max_new_tokens=25)
            caps = processor.batch_decode(out, skip_special_tokens=True)
            for img_id, cap in zip(batch_ids, caps):
                captions_dict[str(img_id)] = cap.strip()

    print(f"[{split_name}] Generated {len(captions_dict)} captions in {time.time() - t0:.1f}s")
    with open(cache_path, "w") as f:
        json.dump(captions_dict, f, indent=2)
    print(f"Saved -> {cache_path}")
    return captions_dict


def extract_augmented_text_features(split_name, csv_path, captions_dict, device):
    out_file = config.CACHE_DIR / f"xlm_roberta_scene_{split_name}.npz"
    if out_file.exists():
        print(f"[{split_name}] Augmented text features already cached at {out_file}")
        return

    df = pd.read_csv(csv_path)
    ids = df["image_id"].astype(int).tolist()

    augmented_texts = []
    for _, row in df.iterrows():
        ocr_clean = str(row["ocr"]).replace("\n", " ").strip() if pd.notna(row["ocr"]) else ""
        caption = captions_dict.get(str(row["image_id"]), "")
        if caption and ocr_clean:
            combined = f"{ocr_clean} | Visual scene: {caption}"
        elif caption:
            combined = f"Visual scene: {caption}"
        elif ocr_clean:
            combined = ocr_clean
        else:
            combined = "."
        augmented_texts.append(combined)

    print(f"\n[{split_name}] Extracting XLM-RoBERTa embeddings for {len(augmented_texts)} scene-augmented texts...")
    tok, text_model = features.load_text_encoder()
    text_model.to(device).eval()

    all_feats = []
    t0 = time.time()
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(augmented_texts), batch_size):
            batch = augmented_texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = text_model(**enc).last_hidden_state[:, 0, :]
            all_feats.append(out.cpu().numpy().astype(np.float32))

    feats = np.concatenate(all_feats, axis=0)
    print(f"Extracted shape: {feats.shape} in {time.time() - t0:.1f}s")
    np.savez_compressed(out_file, ids=np.array(ids), feats=feats)
    print(f"Saved -> {out_file}")


def main():
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"Device: {device} (Apple M5)")

    model_id = "Salesforce/blip-image-captioning-base"
    print(f"Loading {model_id}...")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()

    # 1. Validation split (1,500 memes)
    val_caps = generate_split_captions("val", config.VAL_IMG_DIR, config.VAL_CSV, processor, model, device, batch_size=32)
    extract_augmented_text_features("val", config.VAL_CSV, val_caps, device)

    # 2. Train split (7,000 memes)
    train_caps = generate_split_captions("train", config.TRAIN_IMG_DIR, config.TRAIN_CSV, processor, model, device, batch_size=32)
    extract_augmented_text_features("train", config.TRAIN_CSV, train_caps, device)

    print("\nVisual scene captions & augmented text embeddings successfully generated for all splits!")


if __name__ == "__main__":
    main()
