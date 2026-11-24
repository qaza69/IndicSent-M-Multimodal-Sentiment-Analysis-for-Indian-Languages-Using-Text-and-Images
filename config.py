"""Saari paths aur default settings yahan hain. Zaroorat ho to sirf is file ko edit karo."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Data paths -----------------------------------------------------------
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "updated_train.csv"
VAL_CSV = DATA_DIR / "updated_val.csv"
TRAIN_IMG_DIR = DATA_DIR / "trainImages"   # <- yahan 7000 train images (0.jpg ... 6999.jpg)
VAL_IMG_DIR = DATA_DIR / "valImages"       # <- yahan 1500 val images

# Chhote sample images (repo ke saath aaye the) - sirf quick test ke liye
SAMPLE_TRAIN_IMG_DIR = ROOT / "samples" / "train_images"
SAMPLE_VAL_IMG_DIR = ROOT / "samples" / "val_images"

CACHE_DIR = ROOT / "cache"      # extracted features yahan save hote hain (dobara fast chalne ke liye)
HF_HOME = CACHE_DIR / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
OUTPUT_DIR = ROOT / "outputs"   # results, plots, trained model yahan milenge

# ---- Models ---------------------------------------------------------------
# Text encoder: XLM-RoBERTa for superior cross-lingual & code-mixed Hinglish representation
# (Twitter-RoBERTa tested but underperformed on Hinglish OCR text — XLM-R handles multilingual better)
TEXT_MODEL_NAME = "xlm-roberta-base"
# Vision encoder: CLIP ViT-B/32 (512-dim) for contextual multimodal meme understanding
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_VISION_BACKBONE = "clip"
MAX_SEQ_LEN = 128
IMG_SIZE = 224
SEED = 42

# ---- Labels ---------------------------------------------------------------
TASK_A_LABELS = ["negative", "neutral", "positive"]
TASK_B_LABELS = ["humour", "sarcastic", "offensive", "motivational"]

# ---- Training defaults -------------------
DEFAULTS = {
    ("text", "A"):       dict(epochs=25, batch_size=32, lr=2e-4),
    ("text", "B"):       dict(epochs=25, batch_size=32, lr=2e-4),
    ("image", "A"):      dict(epochs=30, batch_size=32, lr=1e-4),
    ("image", "B"):      dict(epochs=40, batch_size=32, lr=1e-4),
    ("multimodal", "A"): dict(epochs=25, batch_size=32, lr=2e-4),
    ("multimodal", "B"): dict(epochs=25, batch_size=16, lr=5e-4),
}
