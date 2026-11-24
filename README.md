# IndicSent-M: Multimodal Sentiment Analysis for Indian Languages Using Text and Images

IndicSent-M classifies the sentiment and emotions of memes by combining their text and images. It currently targets Hinglish (Hindi-English code-mixed) memes, using multilingual Indic encoders (XLM-RoBERTa, MuRIL). It is built with PyTorch and HuggingFace Transformers and includes a local web studio for live predictions.

- **Task A, sentiment:** negative / neutral / positive
- **Task B, emotions (multi-label):** humour, sarcastic, offensive, motivational

## What's inside

| Area | Techniques |
|---|---|
| Text encoders | XLM-RoBERTa-base, Google MuRIL, Hinglish text normalization (slang, elongation, emoji expansion) |
| Vision encoders | CLIP ViT-B/32 (and ViT-L), VGG16 baseline, BLIP scene captions merged into the text |
| Fusion | Squeeze-and-Excitation modality gating, text/image interaction features, cross-attention, emotion-conditioned sentiment head |
| Training | Multi-task (A + B), focal loss, ordinal loss with label smoothing, LoRA fine-tuning, SentiMix pre-training, Memotion 1 augmentation |
| Post-processing | Probability calibration, per-emotion thresholds, model ensembles |
| App | `app.py`: a local web UI and JSON API (`/api/predict`, `/api/status`, `/api/metrics`) |

Pre-trained backbone features are computed once and cached in `cache/`, so the classification heads train in minutes, even on a CPU.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support, install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/.

## Data (not included)

The datasets belong to their creators and are not redistributed here.

- **Memotion 3** (https://aiisc.ai/defactify2/Memotion.html): place the annotation CSVs at `data/updated_train.csv` and `data/updated_val.csv`, and the images at `data/trainImages/0.jpg … 6999.jpg` and `data/valImages/0.jpg … 1499.jpg`.
- **SentiMix (SemEval-2020 Task 9)**, optional: see `prepare_sentimix.py`.
- **Memotion 1**, optional: `download_and_extract_memotion1.py`.

Check your setup with `python train.py --check`.

## Usage

```bash
# baselines
python train.py --model text       --task A
python train.py --model multimodal --task B

# advanced pipelines
python train_task_a_normalized_sota.py
python train_task_b_v2.py
python evaluate_task_b_best.py

# predict a single meme
python predict.py --model multimodal --task A --image my_meme.jpg --text "meme ka text"

# web studio
python app.py --port 8501        # open http://localhost:8501
```

Results are saved to `outputs/<experiment>/`: `metrics.json`, `model.pt` and training plots.

## Results (Memotion 3 validation set, 1,500 memes)

| Task | Model | Metric | Score |
|---|---|---|---|
| A | Text baseline (XLM-R) | macro-F1 | 0.35 |
| A | Dual-text + CLIP + emotion gate, using *predicted* emotions | macro-F1 | 0.34 |
| B | SE-fusion ensemble with per-emotion thresholds | weighted-F1 / macro-F1 | 0.815 / 0.533 |

## Acknowledgements

- Pretrained models: XLM-RoBERTa (Meta AI), MuRIL (Google), CLIP (OpenAI), BLIP (Salesforce), VGG16 (torchvision).
- Datasets: Memotion 3, Memotion 1, SentiMix.
- Parts of the early baseline code derive from MIT-licensed work; see `LICENSE.txt`.

## License

MIT; see `LICENSE.txt`.
