"""Trainable classification heads with LayerNorm, Dropout, and balanced projection.

- text      : XLM-R [CLS] (768)  -> LayerNorm -> Dropout(0.3) -> Linear(256) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(classes)
- image     : VGG16 (25088)      -> LayerNorm -> Dropout(0.5) -> Linear(512) -> LayerNorm -> ReLU -> Dropout(0.4) -> Linear(classes)
- multimodal: txt(256) + img(256) balanced projection -> LayerNorm -> Concat(512) -> Linear(128) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(classes)

Output: raw logits (Task A: CrossEntropyLoss, Task B: BCEWithLogitsLoss).
"""
import torch
from torch import nn

from features import TEXT_FEAT_DIM, VGG_FEAT_DIM, CLIP_FEAT_DIM


class TextHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(TEXT_FEAT_DIM),
            nn.Dropout(0.3),
            nn.Linear(TEXT_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_out),
        )

    def forward(self, txt):
        return self.net(txt)


class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(VGG_FEAT_DIM),
            nn.Dropout(0.5),
            nn.Linear(VGG_FEAT_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_out),
        )

    def forward(self, img):
        return self.net(img)


class MultimodalHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        # Balance dimensionality: project 25088 visual features down to 256, and 768 text down to 256
        self.img = nn.Sequential(
            nn.LayerNorm(VGG_FEAT_DIM),
            nn.Dropout(0.5),
            nn.Linear(VGG_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.txt = nn.Sequential(
            nn.LayerNorm(TEXT_FEAT_DIM),
            nn.Dropout(0.3),
            nn.Linear(TEXT_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Fuse balanced 512-dim features
        self.fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_out),
        )

    def forward(self, txt, img):
        t = self.txt(txt)
        i = self.img(img)
        return self.fusion(torch.cat([t, i], dim=1))


class CLIPImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(CLIP_FEAT_DIM),
            nn.Dropout(0.3),
            nn.Linear(CLIP_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_out),
        )

    def forward(self, img):
        return self.net(img)


class CLIPMultimodalHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.img = nn.Sequential(
            nn.LayerNorm(CLIP_FEAT_DIM),
            nn.Dropout(0.3),
            nn.Linear(CLIP_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.txt = nn.Sequential(
            nn.LayerNorm(TEXT_FEAT_DIM),
            nn.Dropout(0.3),
            nn.Linear(TEXT_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Incongruity & Interaction: [t; i; t*i; |t-i|; cos_sim] = 256*4 + 1 = 1025 dims
        self.fusion = nn.Sequential(
            nn.Linear(1025, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
        )

    def forward(self, txt, img):
        t = self.txt(txt)
        i = self.img(img)
        prod = t * i
        diff = torch.abs(t - i)
        sim = torch.cosine_similarity(t, i, dim=1, eps=1e-6).unsqueeze(1)
        fused = torch.cat([t, i, prod, diff, sim], dim=1)
        return self.fusion(fused)


class SEFusionMultimodalHead(nn.Module):
    """Squeeze-and-Excitation Multimodal Fusion.

    Inspired by NUAA-QMUL-AIIT (1st Place in Memotion 3 Task A).
    Learns dynamic modality-specific importance weights to adaptively combine
    textual and visual features based on meme context.
    """
    def __init__(self, n_out, vision_dim=CLIP_FEAT_DIM, hidden_dim=256):
        super().__init__()
        self.txt = nn.Sequential(
            nn.LayerNorm(TEXT_FEAT_DIM),
            nn.Dropout(0.2),
            nn.Linear(TEXT_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.img = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(0.2),
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        # Squeeze-and-Excitation gating: predicts [w_text, w_image]
        self.se_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=1),
        )
        # Deep fusion classifier with residual representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
        )

    def forward(self, txt, img):
        t = self.txt(txt)   # [B, hidden_dim]
        i = self.img(img)   # [B, hidden_dim]
        # Dynamic modality gating
        weights = self.se_gate(torch.cat([t, i], dim=1))  # [B, 2]
        w_t = weights[:, 0:1]
        w_i = weights[:, 1:2]
        gated = w_t * t + w_i * i
        # Multi-branch fusion
        fused = torch.cat([gated, t, i], dim=1)
        return self.classifier(fused)


class MultiTaskSEFusionHead(nn.Module):
    """Joint Multi-Task Head for Task A (Sentiment, 3 classes) and Task B (Emotions, 4 binary classes).

    Features:
    1. Squeeze-and-Excitation multimodal fusion of XLM-R and CLIP.
    2. Shared cross-modal representation.
    3. Task B emotion head predicts: humour, sarcasm, offensive, motivational.
    4. Task A sentiment head is conditioned on BOTH shared features AND predicted emotion logits,
       allowing the network to use emotion cues (e.g. sarcasm, offense) to resolve sentiment polarity.
    """
    def __init__(self, vision_dim=CLIP_FEAT_DIM, hidden_dim=256):
        super().__init__()
        self.txt = nn.Sequential(
            nn.LayerNorm(TEXT_FEAT_DIM),
            nn.Dropout(0.2),
            nn.Linear(TEXT_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.img = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(0.2),
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.se_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=1),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        # Task B emotion head (4 classes)
        self.head_b = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        # Task A sentiment head (3 classes, conditioned on shared + emotion features)
        self.head_a = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def forward(self, txt, img):
        t = self.txt(txt)
        i = self.img(img)
        weights = self.se_gate(torch.cat([t, i], dim=1))
        w_t = weights[:, 0:1]
        w_i = weights[:, 1:2]
        gated = w_t * t + w_i * i
        fused = torch.cat([gated, t, i], dim=1)
        shared_feat = self.shared(fused)

        logits_b = self.head_b(shared_feat)
        logits_a = self.head_a(torch.cat([shared_feat, logits_b], dim=1))
        return logits_a, logits_b


def build_head(model: str, task: str = "A", vision_backbone: str = "clip", fusion: str = "se") -> nn.Module:
    if model == "multitask":
        vision_dim = CLIP_FEAT_DIM if vision_backbone == "clip" else VGG_FEAT_DIM
        return MultiTaskSEFusionHead(vision_dim=vision_dim)
    n_out = 3 if task == "A" else 4
    if model == "text":
        return TextHead(n_out)
    if model == "image":
        return CLIPImageHead(n_out) if vision_backbone == "clip" else ImageHead(n_out)
    # Multimodal:
    vision_dim = CLIP_FEAT_DIM if vision_backbone == "clip" else VGG_FEAT_DIM
    if fusion == "se":
        return SEFusionMultimodalHead(n_out, vision_dim=vision_dim)
    elif vision_backbone == "clip":
        return CLIPMultimodalHead(n_out)
    else:
        return MultimodalHead(n_out)


# -----------------------------------------------------------------------------
# Legacy Classification Heads (for backward compatibility with older checkpoints)
# -----------------------------------------------------------------------------
class LegacyTextHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_FEAT_DIM, TEXT_FEAT_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(TEXT_FEAT_DIM, n_out),
        )

    def forward(self, txt):
        return self.net(txt)


class LegacyImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VGG_FEAT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_out),
        )

    def forward(self, img):
        return self.net(img)


class LegacyMultimodalHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.txt = nn.Sequential(
            nn.Linear(TEXT_FEAT_DIM, TEXT_FEAT_DIM),
            nn.ReLU(),
            nn.Linear(TEXT_FEAT_DIM, 512),
            nn.ReLU(),
        )
        self.img = nn.Sequential(
            nn.Linear(VGG_FEAT_DIM, 512),
            nn.ReLU(),
        )
        self.out = nn.Linear(1024, n_out)

    def forward(self, txt, img):
        return self.out(torch.cat([self.txt(txt), self.img(img)], dim=1))


def load_head(checkpoint_data: dict, model: str, task: str = "A") -> nn.Module:
    """Load model head with automatic architecture detection.

    Supports:
    1. MultiTask architecture
    2. CLIP architecture (512-dim vision)
    3. Modern VGG architecture (25088-dim vision with LayerNorm)
    4. Legacy VGG architecture (without LayerNorm)
    """
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    if model == "multitask" or checkpoint_data.get("model") == "multitask":
        head = MultiTaskSEFusionHead()
        head.load_state_dict(state_dict)
        return head
    backbone = checkpoint_data.get("vision_backbone", None)

    candidates = ["vgg16", "clip"] if backbone == "vgg16" else ["clip", "vgg16"]
    for b in candidates:
        for f in ["se", "clip"]:
            try:
                head = build_head(model, task, vision_backbone=b, fusion=f)
                head.load_state_dict(state_dict)
                return head
            except (RuntimeError, KeyError):
                continue

    n_out = 3 if task == "A" else 4
    # Fall back to legacy architecture for older checkpoints
    legacy_classes = {
        "text": LegacyTextHead,
        "image": LegacyImageHead,
        "multimodal": LegacyMultimodalHead,
    }
    legacy_head = legacy_classes[model](n_out)
    legacy_head.load_state_dict(state_dict)
    return legacy_head

