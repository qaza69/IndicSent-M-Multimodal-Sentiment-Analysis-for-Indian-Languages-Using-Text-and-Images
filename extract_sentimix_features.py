"""Extract and cache XLM-RoBERTa features for SemEval 2020 Task 9 (SentiMix Hinglish)."""
import numpy as np
import pandas as pd
import torch

import config
import features


def main():
    device = features.get_device()
    print(f"Using device: {device}")
    
    train_csv = config.DATA_DIR / "sentimix_train.csv"
    dev_csv = config.DATA_DIR / "sentimix_dev.csv"

    df_train = pd.read_csv(train_csv)
    df_dev = pd.read_csv(dev_csv)

    tok, enc = features.load_text_encoder()

    cache_train = config.CACHE_DIR / "sentimix_xlm_roberta_train.npz"
    if not cache_train.exists():
        print(f"Extracting features for {len(df_train)} train samples...")
        feats_train = features.compute_text(df_train["text"].tolist(), device, batch_size=64, encoder=(tok, enc))
        np.savez(cache_train, feats=feats_train, labels=df_train["label"].values)
        print(f"Saved: {feats_train.shape} -> {cache_train}")
    else:
        print(f"Train features already cached at {cache_train}")

    cache_dev = config.CACHE_DIR / "sentimix_xlm_roberta_dev.npz"
    if not cache_dev.exists():
        print(f"Extracting features for {len(df_dev)} dev samples...")
        feats_dev = features.compute_text(df_dev["text"].tolist(), device, batch_size=64, encoder=(tok, enc))
        np.savez(cache_dev, feats=feats_dev, labels=df_dev["label"].values)
        print(f"Saved: {feats_dev.shape} -> {cache_dev}")
    else:
        print(f"Dev features already cached at {cache_dev}")

    print("Done!")


if __name__ == "__main__":
    main()
