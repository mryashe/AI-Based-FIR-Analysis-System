"""
Embeddings + Similarity Search for BNS sections.

Install:  pip install sentence-transformers numpy
Run:      python embedding_search.py

IMPORTANT: The sample data below uses placeholders for section numbers and
punishments. Fill bns_sections.json from the OFFICIAL BNS 2023 text (India Code
or the Ministry of Home Affairs website), not from memory.
"""
import hashlib
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_FILE = "bns_sections.json"
CACHE_FILE = "section_embeddings.npz"

SAMPLE_SECTIONS = [
    {
        "section": "FILL_FROM_OFFICIAL_BNS",
        "title": "Theft",
        "description": "Dishonestly taking movable property out of someone's possession without their consent.",
        "punishment": "FILL_FROM_OFFICIAL_BNS",
        "keywords": ["stole", "stolen", "pickpocket", "snatched", "jewellery", "चोरी", "चोरले", "दागिने"],
    },
    {
        "section": "FILL_FROM_OFFICIAL_BNS",
        "title": "House-breaking by night",
        "description": "Entering or breaking into a house after sunset and before sunrise to commit an offence.",
        "punishment": "FILL_FROM_OFFICIAL_BNS",
        "keywords": ["broke into house", "night", "burglary", "घरफोडी", "रात्री", "घुसले"],
    },
    {
        "section": "FILL_FROM_OFFICIAL_BNS",
        "title": "Voluntarily causing hurt",
        "description": "Intentionally causing bodily pain, disease or infirmity to another person.",
        "punishment": "FILL_FROM_OFFICIAL_BNS",
        "keywords": ["beat", "hit", "slapped", "attacked", "मारहाण", "मारले"],
    },
    {
        "section": "FILL_FROM_OFFICIAL_BNS",
        "title": "Cheating",
        "description": "Deceiving a person to dishonestly make them hand over property or money.",
        "punishment": "FILL_FROM_OFFICIAL_BNS",
        "keywords": ["fraud", "cheated", "fake", "online scam", "फसवणूक", "फसवले"],
    },
]


def load_sections():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_SECTIONS, f, ensure_ascii=False, indent=2)
        print(f"Created sample {DATA_FILE}. Replace it with real BNS data.")
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_text(section):
    """The text we embed for each section: title + description + everyday keywords."""
    keywords = ", ".join(section.get("keywords", []))
    return f"{section['title']}. {section['description']} {keywords}"


def load_or_build_embeddings(model, sections):
    """Embed all sections once and cache them; re-embed only if the data changed."""
    texts = [build_text(s) for s in sections]
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()

    if os.path.exists(CACHE_FILE):
        cache = np.load(CACHE_FILE, allow_pickle=False)
        if str(cache["digest"]) == digest:
            return cache["embeddings"]

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.savez(CACHE_FILE, embeddings=embeddings, digest=digest)
    return embeddings


def suggest(model, sections, section_emb, complaint, k=5, min_score=0.30):
    """Return up to k sections whose similarity to the complaint is >= min_score."""
    query_emb = model.encode(complaint, normalize_embeddings=True)
    scores = section_emb @ query_emb  # cosine similarity (vectors are normalised)
    top_idx = np.argsort(scores)[::-1][:k]
    return [
        {**sections[i], "score": round(float(scores[i]), 3)}
        for i in top_idx
        if scores[i] >= min_score
    ]


def explain(model, section_emb_row, complaint, top_n=5):
    """Occlusion XAI: remove each word and see how much the similarity drops."""
    words = complaint.split()
    if len(words) < 2:
        return []
    base = float(section_emb_row @ model.encode(complaint, normalize_embeddings=True))
    reduced_texts = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    reduced_emb = model.encode(reduced_texts, normalize_embeddings=True)
    drops = base - (reduced_emb @ section_emb_row)
    ranked = sorted(zip(words, drops), key=lambda x: -x[1])[:top_n]
    return [(w, round(float(d), 3)) for w, d in ranked]


if __name__ == "__main__":
    print("Loading model (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)
    sections = load_sections()
    section_emb = load_or_build_embeddings(model, sections)

    complaints = [
        "A person broke into my house at night and stole jewellery.",
        "रात्री माझ्या घरात कोणीतरी घुसले आणि दागिने चोरून नेले.",
        "He cheated me of 50,000 rupees by a fake online offer.",
    ]

    for text in complaints:
        print("\nComplaint:", text)
        results = suggest(model, sections, section_emb, text)
        if not results:
            print("  No section crossed the score threshold.")
        for r in results:
            idx = sections.index({k: v for k, v in r.items() if k != "score"})
            why = explain(model, section_emb[idx], text)
            print(f"  [{r['score']}] {r['title']} (Sec {r['section']})")
            print(f"        key words: {why}")
