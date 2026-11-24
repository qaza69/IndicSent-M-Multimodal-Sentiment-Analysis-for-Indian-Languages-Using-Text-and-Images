"""Extract MuRIL features on Hinglish-Normalized texts.

Saves:
- cache/muril_normalized_train.npz
- cache/muril_normalized_val.npz
"""
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import config
import features

os.environ["HF_HOME"] = str(config.HF_HOME)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "google/muril-base-cased"


def extract_normalized(split_name, tok, model, device, batch_size=64):
    print(f"\n--- Extracting MuRIL Normalized Features for {split_name} ---")
    json_path = config.CACHE_DIR / f"normalized_texts_{split_name}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    ids = data_json["ids"]
    texts = data_json["texts"]

    model = model.to(device).eval()
    all_feats = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"MuRIL {split_name}", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [t if isinstance(t, str) and t.strip() else " " for t in batch_texts]
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model(**enc)
            cls_feats = out.last_hidden_state[:, 0, :].half().cpu().numpy()
            all_feats.append(cls_feats)

    feats_np = np.concatenate(all_feats, axis=0)
    out_file = config.CACHE_DIR / f"muril_normalized_{split_name}.npz"
    np.savez(out_file, ids=np.array(ids, dtype=np.int64), feats=feats_np)
    print(f"Saved {feats_np.shape} -> {out_file}")
    return feats_np


def main():
    device = features.get_device()
    snapshot = features._find_snapshot(MODEL_NAME)
    source = snapshot if snapshot else MODEL_NAME
    print(f"Loading MuRIL from: {source}")
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)

    extract_normalized("train", tok, model, device)
    extract_normalized("val", tok, model, device)
    print("\nMuRIL Normalized feature extraction complete!")


if __name__ == "__main__":
    main()
