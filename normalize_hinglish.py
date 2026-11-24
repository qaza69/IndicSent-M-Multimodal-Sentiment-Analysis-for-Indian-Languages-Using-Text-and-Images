"""Hinglish Text Normalization & Emoji Expansion Pipeline for Memotion 3.

Processes raw OCR text:
1. Elongation reduction: collapses character repetitions (e.g. 'bhaaaaiiii' -> 'bhai').
2. Slang & chat contraction normalization (e.g. 'nhiii' -> 'nahi', 'bhot' -> 'bahut', 'nibba', 'dank', 'op').
3. Emoji expansion to affective semantic tokens (e.g. 😂 -> [laughing with tears], 🙄 -> [face with rolling eyes]).
4. Whitespace and punctuation cleanup.

Saves normalized text cache to:
- cache/normalized_texts_train.json
- cache/normalized_texts_val.json
"""
import re
import json
import unicodedata
from typing import Dict, Tuple, List, Any
import data
import config


HINGLISH_LEXICON: Dict[str, str] = {
    "nhi": "nahi", "nhn": "nahi", "nhiii": "nahi", "nahin": "nahi", "nai": "nahi",
    "bcz": "because", "coz": "because", "cuz": "because", "bcuz": "because",
    "bhai": "brother", "bhaii": "brother", "bhaiii": "brother", "bro": "brother",
    "yaar": "friend", "yaaar": "friend", "dost": "friend", "dosti": "friendship",
    "kya": "what", "kyu": "why", "kyun": "why", "kyuki": "because", "kyunki": "because",
    "kaisa": "how", "kaisi": "how", "kaise": "how",
    "accha": "good", "achha": "good", "acha": "good", "acchi": "good", "acche": "good",
    "bahut": "very", "bht": "very", "bohot": "very", "bhot": "very",
    "bekaar": "useless", "bekar": "useless", "bakwaas": "nonsense", "bakwas": "nonsense",
    "shandar": "magnificent", "badhiya": "great", "jhakas": "awesome", "fadu": "awesome",
    "dhansu": "terrific", "mast": "great", "mazak": "joke", "mzaak": "joke",
    "pyaar": "love", "pyar": "love", "mohabbat": "love", "ishq": "love",
    "dil": "heart", "dukh": "sadness", "dard": "pain", "rona": "cry",
    "gussa": "anger", "hasna": "laugh", "khush": "happy", "khushi": "happiness",
    "desh": "country", "log": "people", "sab": "everyone", "apna": "ours",
    "kuch": "something", "kuchh": "something", "koi": "anyone",
    "ab": "now", "kab": "when", "tab": "then", "aaj": "today", "kal": "tomorrow",
    "chal": "walk/go", "chalo": "let's go", "nikal": "leave",
    "nibba": "naive immature boy", "nibbi": "naive immature girl",
    "chapri": "cringe flashy youth", "dank": "edgy humor",
    "normie": "conventional person", "bhakt": "devotee/partisan",
    "andhbhakt": "blind supporter", "libbu": "liberal caricature",
    "chaddi": "right-wing slang", "chutiya": "foolish person",
    "kamina": "rascal", "pagal": "crazy", "harami": "mischievous",
    "op": "overpowered", "ded": "extremely funny/laughing",
    "lmao": "laughing heartily", "rofl": "rolling laughing",
    "rip": "rest in peace", "lit": "exciting", "fam": "friends",
    "turu": "true", "lob": "love", "tru": "true", "khtm": "finished",
    "tata": "goodbye", "byebye": "goodbye", "alvida": "farewell",
    "rok": "stop", "sahi": "correct", "galat": "wrong"
}

EMOJI_SENTIMENT_MAP: Dict[str, Tuple[str, str]] = {
    "😂": ("laughing with tears", "positive"),
    "🤣": ("rolling laughing", "positive"),
    "😭": ("loudly crying", "negative"),
    "🥺": ("pleading face", "neutral"),
    "😡": ("pouting angry", "negative"),
    "🤬": ("cursing angry", "negative"),
    "❤️": ("red heart", "positive"),
    "🔥": ("fire lit", "positive"),
    "💀": ("skull dead laughing", "humour"),
    "🤡": ("clown foolish", "sarcastic"),
    "👍": ("thumbs up", "positive"),
    "👎": ("thumbs down", "negative"),
    "🙏": ("folded hands respect", "positive"),
    "😏": ("smirking sarcastic", "sarcastic"),
    "🙄": ("face with rolling eyes", "sarcastic"),
    "🤦": ("person facepalming", "negative"),
    "🤢": ("nauseated face", "negative"),
    "💩": ("pile of poo", "negative"),
    "😎": ("cool smiling face", "positive"),
    "💔": ("broken heart", "negative")
}


def reduce_elongated_words(word: str) -> str:
    """Collapses character repetition >= 3 to 1 or 2 (e.g. bhaaaai -> bhai)."""
    return re.sub(r'(.)\1{2,}', r'\1\1', word)


def extract_emojis(text: str) -> str:
    """Expands affective emojis into semantic tokens."""
    result_chars = []
    for ch in text:
        if ch in EMOJI_SENTIMENT_MAP:
            desc, _ = EMOJI_SENTIMENT_MAP[ch]
            result_chars.append(f" [{desc}] ")
        else:
            result_chars.append(ch)
    return "".join(result_chars)


def normalize_hinglish_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalization
    text_norm = unicodedata.normalize("NFKD", text)

    # 2. Expand emojis
    text_norm = extract_emojis(text_norm)

    # 3. Clean linebreaks & excess spaces
    text_norm = re.sub(r'[\r\n\t]+', ' ', text_norm)
    text_norm = re.sub(r'\s+', ' ', text_norm).strip()

    # 4. Token normalization
    tokens = text_norm.split()
    normalized_tokens = []

    for token in tokens:
        clean_token = re.sub(r'^[^\w]+|[^\w]+$', '', token.lower())
        reduced_token = reduce_elongated_words(clean_token)

        if reduced_token in HINGLISH_LEXICON:
            normalized_tokens.append(reduced_token)
        elif clean_token in HINGLISH_LEXICON:
            normalized_tokens.append(clean_token)
        else:
            if len(token) > 0 and token.lower() != reduced_token:
                normalized_tokens.append(reduced_token)
            else:
                normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def main():
    print("=" * 65)
    print("RUNNING HINGLISH TEXT NORMALIZATION & EMOJI EXPANSION PASS")
    print("=" * 65)

    for split in ["train", "val"]:
        s = data.load_split(split, "A", "multimodal", samples=False, balance=False)
        print(f"Normalizing {split} split ({len(s.texts)} texts)...")

        normalized = []
        for t in s.texts:
            norm_t = normalize_hinglish_text(t)
            normalized.append(norm_t)

        out_path = config.CACHE_DIR / f"normalized_texts_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"ids": [int(i) for i in s.ids], "texts": normalized}, f, ensure_ascii=False, indent=2)

        print(f"Saved {split} normalized texts -> {out_path}")
        print("Sample raw vs normalized:")
        for k in range(min(3, len(s.texts))):
            print(f"  Raw:        {s.texts[k][:70]}")
            print(f"  Normalized: {normalized[k][:70]}\n")


if __name__ == "__main__":
    main()
