"""Download and parse SemEval-2020 Task 9 (SentiMix Hinglish) dataset.

Extracts:
  - data/sentimix_train.csv (14,000 samples)
  - data/sentimix_dev.csv   (3,000 samples)

Labels:
  negative -> 0
  neutral  -> 1
  positive -> 2
"""
import re
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

import config

REPO_ID = "RTT1/SentiMix"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def parse_conll(filepath: str) -> list[dict]:
    samples = []
    curr_tokens = []
    curr_meta = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_meta and curr_tokens:
                    text = " ".join(curr_tokens)
                    # Clean up URL tokens and spaced punctuation
                    text = re.sub(r"https? // \S+", "", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text and curr_meta[1] in LABEL_MAP:
                        samples.append({
                            "uid": curr_meta[0],
                            "text": text,
                            "sentiment": curr_meta[1],
                            "label": LABEL_MAP[curr_meta[1]],
                        })
                curr_tokens = []
                curr_meta = None
                continue

            if line.startswith("meta\t"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    curr_meta = (parts[1], parts[2].lower().strip())
            else:
                parts = line.split("\t")
                if parts:
                    curr_tokens.append(parts[0])

        # Final record
        if curr_meta and curr_tokens:
            text = " ".join(curr_tokens)
            text = re.sub(r"https? // \S+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text and curr_meta[1] in LABEL_MAP:
                samples.append({
                    "uid": curr_meta[0],
                    "text": text,
                    "sentiment": curr_meta[1],
                    "label": LABEL_MAP[curr_meta[1]],
                })

    return samples


def main():
    print("[SentiMix] Downloading files from HuggingFace...")
    p_train = hf_hub_download(repo_id=REPO_ID, filename="train_14k_split_conll.txt", repo_type="dataset")
    p_dev = hf_hub_download(repo_id=REPO_ID, filename="dev_3k_split_conll.txt", repo_type="dataset")

    print("[SentiMix] Parsing train split...")
    train_data = parse_conll(p_train)
    df_train = pd.DataFrame(train_data)
    out_train = config.DATA_DIR / "sentimix_train.csv"
    df_train.to_csv(out_train, index=False)
    print(f"Saved {len(df_train)} train samples -> {out_train}")
    print(df_train["sentiment"].value_counts())

    print("\n[SentiMix] Parsing dev split...")
    dev_data = parse_conll(p_dev)
    df_dev = pd.DataFrame(dev_data)
    out_dev = config.DATA_DIR / "sentimix_dev.csv"
    df_dev.to_csv(out_dev, index=False)
    print(f"Saved {len(df_dev)} dev samples -> {out_dev}")
    print(df_dev["sentiment"].value_counts())


if __name__ == "__main__":
    main()
