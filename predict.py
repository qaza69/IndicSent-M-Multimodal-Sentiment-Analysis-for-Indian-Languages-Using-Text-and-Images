"""Trained model se ek meme ko classify karo.

Examples
--------
python predict.py --model multimodal --task A --image samples/train_images/326.jpg --text "Tum paas aaye yun muskuraye"
python predict.py --model image      --task B --image my_meme.jpg
python predict.py --model text       --task A --text "bhai ye toh bahut mast hai"

Pehle `python train.py --model <same model> --task <same task>` chala ke model train kar lo.
"""
import argparse

import numpy as np
import torch

import config
import features
from models import build_head, load_head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["text", "image", "multimodal", "sota"])
    p.add_argument("--task", required=True, choices=["A", "B"])
    p.add_argument("--image", help="meme image ka path")
    p.add_argument("--text", help="meme ke andar likha text (OCR)")
    p.add_argument("--samples", action="store_true", help="samples wala trained model use karo")
    p.add_argument("--lora", action="store_true", help="use lora fine-tuned checkpoint")
    args = p.parse_args()

    if args.model in ("image", "multimodal", "sota") and not args.image:
        p.error("is model ke liye --image chahiye")
    if args.model in ("text", "multimodal", "sota") and not args.text:
        p.error("is model ke liye --text chahiye")

    if args.model == "sota":
        import app
        res = app.predict_meme("sota", args.task, text=args.text, image_path=args.image)
        if args.task == "A":
            print("\nSentiment prediction (SOTA Ensemble):")
            for name in config.TASK_A_LABELS:
                pr = res["probs"][name]
                print(f"  {name:<9} {pr:6.1%}")
            print(f"=> {res['best_label'].upper()} ({res['confidence']*100:.1f}%)")
            print(f"Architecture: {res['architecture']}")
        else:
            print("\nEmotion prediction (SOTA Ensemble):")
            for name in config.TASK_B_LABELS:
                em = res["emotions"][name]
                flag = "YES" if em["detected"] else "no"
                print(f"  {name:<13} {em['confidence']:6.1%}  {flag}")
            print(f"=> {res['best_label']}")
            print(f"Architecture: {res['architecture']}")
        return

    suffix = "_lora" if args.lora else ("_samples" if args.samples else "")
    tag = f"{args.model}_task{args.task}" + suffix
    ckpt_path = config.OUTPUT_DIR / tag / "model.pt"
    if not ckpt_path.exists():
        # Try fallback
        alt_tag = f"{args.model}_task{args.task}"
        if (config.OUTPUT_DIR / alt_tag / "model.pt").exists():
            ckpt_path = config.OUTPUT_DIR / alt_tag / "model.pt"
            tag = alt_tag
        else:
            raise SystemExit(f"Trained model nahi mila: {ckpt_path}\nPehle train.py chalao.")

    device = features.get_device()
    checkpoint_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = load_head(checkpoint_dict, args.model, args.task)
    head.to(device).eval()

    xs = []
    if args.model in ("text", "multimodal"):
        lora_dir = ckpt_path.parent / "lora_adapter"
        encoder = None
        if lora_dir.exists():
            from peft import PeftModel
            tok, enc = features.load_text_encoder()
            enc = PeftModel.from_pretrained(enc, str(lora_dir))
            encoder = (tok, enc)
        xs.append(features.compute_text([args.text.replace("\n", " ")], device, encoder=encoder))
    if args.model in ("image", "multimodal"):
        from models import CLIPImageHead, CLIPMultimodalHead, SEFusionMultimodalHead, MultiTaskSEFusionHead
        if isinstance(head, (CLIPImageHead, CLIPMultimodalHead, SEFusionMultimodalHead, MultiTaskSEFusionHead)):
            xs.append(features.compute_clip([args.image], device))
        else:
            xs.append(features.compute_vgg([args.image], device, workers=0))
    xs = [torch.from_numpy(x).float().to(device) for x in xs]

    with torch.no_grad():
        out = head(*xs)
        if isinstance(head, MultiTaskSEFusionHead):
            logits = (out[0] if args.task == "A" else out[1])[0].cpu()
        else:
            logits = out[0].cpu()

    if args.task == "A":
        probs = torch.softmax(logits, dim=0).numpy()
        print("\nSentiment prediction:")
        for name, pr in zip(config.TASK_A_LABELS, probs):
            print(f"  {name:<9} {pr:6.1%}")
        print(f"=> {config.TASK_A_LABELS[int(np.argmax(probs))].upper()}")
    else:
        probs = torch.sigmoid(logits).numpy()
        print("\nEmotion prediction:")
        for name, pr in zip(config.TASK_B_LABELS, probs):
            print(f"  {name:<13} {pr:6.1%}  {'YES' if pr > 0.5 else 'no'}")


if __name__ == "__main__":
    main()
