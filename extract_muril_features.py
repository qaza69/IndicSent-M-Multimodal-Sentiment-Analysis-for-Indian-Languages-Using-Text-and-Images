"""Extract MuRIL (Multilingual Representations for Indian Languages) text features for Memotion 3.

MuRIL is pre-trained by Google on 17 Indian languages and English, including
transliterated texts (Hinglish). This makes its tokenization and semantic representations
significantly better suited for Hinglish meme texts than generic multilingual XLM-R.

Saves:
- cache/google_muril_base_cased_train.npz (ids, feats)
- cache/google_muril_base_cased_val.npz   (ids, feats)
"""
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import config
import data
import features

os.environ["HF_HOME"] = str(config.HF_HOME)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "google/muril-base-cased"


def get_muril():
    local_dir = features._find_snapshot(MODEL_NAME)
    source = local_dir if local_dir else MODEL_NAME
    print(f"Loading MuRIL from: {source}")
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    return tok, model


@torch.no_grad()
def extract_split(split_name, tok, model, device, batch_size=64):
    print(f"\n--- Extracting MuRIL features for {split_name} split ---")
    s = data.load_split(split_name, "A", "multimodal", samples=False, balance=False)
    ids = s.ids
    texts = s.texts

    model = model.to(device).eval()
    all_feats = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"MuRIL {split_name}", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        # Replace empty strings with a space or default token
        batch_texts = [t if isinstance(t, str) and t.strip() else " " for t in batch_texts]
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            return_tensors="pt"
        ).to(device)

        out = model(**enc)
        # Extract [CLS] token representation (768-dim)
        cls_feats = out.last_hidden_state[:, 0, :].half().cpu().numpy()
        all_feats.append(cls_feats)

    feats_np = np.concatenate(all_feats, axis=0)
    print(f"Done! Shape: {feats_np.shape}, dtype: {feats_np.dtype}")

    # Save cache
    out_file = config.CACHE_DIR / f"google_muril_base_cased_{split_name}.npz"
    np.savez(out_file, ids=np.array(ids, dtype=np.int64), feats=feats_np)
    print(f"Saved -> {out_file}")
    return feats_np


def main():
    device = features.get_device()
    print(f"Running on device: {device}")
    tok, model = get_muril()
    extract_split("train", tok, model, device)
    extract_split("val", tok, model, device)
    print("\nAll MuRIL features extracted and cached successfully!")


if __name__ == "__main__":
    main()
