"""Download Memotion 1.0 (from Ahren09/MMSoc_Memotion) and precompute CLIP + XLM-R features.

Extracts all 6,992 examples with Task A labels (0: negative, 1: neutral, 2: positive)
and Task B labels (humour, sarcastic, offensive, motivational).
Saves features to disk:
  - cache/memotion1_clip.npz
  - cache/memotion1_xlm_roberta.npz
  - cache/memotion1_labels.npz
"""
import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import torch

import config
import features
from data import TASK_A_MAP, TASK_B_MAPS

REPO_ID = "Ahren09/MMSoc_Memotion"
PARQUET_FILES = [
    "data/test-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]


def load_all_parquets():
    tables = []
    for rel_path in PARQUET_FILES:
        print(f"[Memotion1] Ensuring {rel_path}...")
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset", local_files_only=True)
        except Exception:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        print(f"[Memotion1] Loading table from {local_path}...")
        t = pq.read_table(local_path)
        tables.append(t)
    return tables


def main():
    device = features.get_device()
    print(f"Using device: {device}")
    
    # 1. Download & load tables
    tables = load_all_parquets()
    
    all_images = []
    all_texts = []
    all_y_a = []
    all_y_b = []
    
    print("[Memotion1] Parsing rows...")
    for t in tables:
        df_meta = t.select(["sentiment", "text_corrected", "text_ocr", "humor", "sarcasm", "offensive", "motivational"]).to_pandas()
        img_col = t["image"]
        
        for idx in range(len(df_meta)):
            row = df_meta.iloc[idx]
            sent = str(row["sentiment"]).strip().lower()
            if sent not in TASK_A_MAP:
                continue
            y_a = TASK_A_MAP[sent]
            
            # Text: prefer text_corrected, fall back to text_ocr
            txt = row["text_corrected"] if pd.notna(row["text_corrected"]) and str(row["text_corrected"]).strip() else row["text_ocr"]
            txt = str(txt) if pd.notna(txt) else ""
            
            # Image bytes
            raw_img = img_col[idx].as_py()
            try:
                if isinstance(raw_img, dict) and "bytes" in raw_img:
                    pil_img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
                else:
                    continue
            except Exception:
                continue
            
            # Task B encoding
            b_vec = []
            col_aliases = {"humour": "humor", "sarcastic": "sarcasm", "offensive": "offensive", "motivational": "motivational"}
            for col_name in config.TASK_B_LABELS:
                col_map = TASK_B_MAPS[col_name]
                actual_col = col_aliases.get(col_name, col_name)
                val = str(row[actual_col]).strip().lower()
                b_vec.append(col_map.get(val, 0))
                
            all_images.append(pil_img)
            all_texts.append(txt)
            all_y_a.append(y_a)
            all_y_b.append(b_vec)

    n_samples = len(all_images)
    print(f"[Memotion1] Successfully collected {n_samples} valid image-text pairs.")
    y_a_arr = np.array(all_y_a, dtype=np.int64)
    y_b_arr = np.array(all_y_b, dtype=np.float32)
    
    # 2. Extract CLIP features
    clip_cache = config.CACHE_DIR / "memotion1_clip.npz"
    if not clip_cache.exists():
        print("[Memotion1] Computing CLIP vision embeddings...")
        clip_feats = features.compute_clip(all_images, device, batch_size=32)
        np.savez(clip_cache, feats=clip_feats)
        print(f"[Memotion1] Saved CLIP features: {clip_feats.shape} -> {clip_cache}")
    else:
        print(f"[Memotion1] CLIP cache exists at {clip_cache}")

    # 3. Extract XLM-RoBERTa features
    xlm_cache = config.CACHE_DIR / "memotion1_xlm_roberta.npz"
    if not xlm_cache.exists():
        print("[Memotion1] Computing XLM-RoBERTa text embeddings...")
        xlm_feats = features.compute_text(all_texts, device, batch_size=64)
        np.savez(xlm_cache, feats=xlm_feats)
        print(f"[Memotion1] Saved XLM-RoBERTa features: {xlm_feats.shape} -> {xlm_cache}")
    else:
        print(f"[Memotion1] XLM-RoBERTa cache exists at {xlm_cache}")

    # 4. Save labels & texts
    labels_cache = config.CACHE_DIR / "memotion1_labels.npz"
    np.savez(labels_cache, y_a=y_a_arr, y_b=y_b_arr, texts=all_texts)
    print(f"[Memotion1] Saved labels to {labels_cache}")
    print("Done!")


if __name__ == "__main__":
    main()
