import os
import config
from transformers import AutoTokenizer, AutoModel

os.environ["HF_HOME"] = str(config.HF_HOME)
model_name = "google/muril-base-cased"
print(f"Downloading {model_name} to {config.HF_HOME}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Successfully downloaded MuRIL!")
