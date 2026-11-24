"""Test BLIP image captioning on sample meme images.
"""
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import config

def test():
    device = torch.device("cpu")
    torch.set_num_threads(8)
    print("Loading BLIP captioning model...")
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()

    sample_imgs = list((config.DATA_DIR / "valImages").glob("*.jpg"))[:5]
    print(f"Found {len(sample_imgs)} test images. Generating captions...")

    for p in sample_imgs:
        t0 = time.time()
        img = Image.open(p).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"[{p.name}] ({time.time()-t0:.2f}s): {caption}")

if __name__ == "__main__":
    test()
