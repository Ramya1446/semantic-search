"""
Part 1 – Corpus Preparation, Embedding & Vector Database Setup
===============================================================
Model choice: all-MiniLM-L6-v2
  - 384-dim embeddings, <80 MB, runs on CPU in reasonable time
  - Strong semantic quality on short-to-medium English text
  - Better sentence-level semantics than TF-IDF while staying lightweight

Vector store: ChromaDB (local, persistent, no server required)
  - Supports metadata filtering needed downstream (cluster_id, category)
  - cosine similarity natively, no extra index config

Cleaning decisions (see comments inline):
  - Strip quoted reply lines (">") – they pollute semantics with other people's words
  - Strip email headers (From:, Subject: etc.) – metadata, not content
  - Remove very short documents (<30 tokens after cleaning) – noise only
  - Keep up to 512 word-piece tokens per doc – MiniLM max context is 512
  - No stemming/lemmatisation – sentence-transformers handles sub-word tokenisation
"""

import re
import logging
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CHROMA_PATH = Path("./chroma_db")
COLLECTION_NAME = "newsgroups"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256          # encode in batches to avoid OOM on modest hardware
MAX_WORDS = 300           # truncate to ~300 words; covers 95%+ of doc content
                          # without wasting compute on boilerplate footers

# ── cleaning ───────────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(
    r"^(From|Subject|Organization|Lines|Message-ID|NNTP-Posting-Host|"
    r"Distribution|X-Newsreader|Reply-To|Date|Path|Newsgroups|Summary|"
    r"Keywords|Xref|References|In-Reply-To):.*$",
    re.MULTILINE | re.IGNORECASE,
)
_QUOTE_RE = re.compile(r"^>.*$", re.MULTILINE)          # quoted reply lines
_URL_RE   = re.compile(r"http\S+|www\.\S+")             # bare URLs add noise
_EMAIL_RE = re.compile(r"\S+@\S+")                      # email addresses
_MULTI_SPACE = re.compile(r"\s{2,}")


def clean(text: str) -> str:
    """Return a cleaned version of a newsgroup post."""
    text = _HEADER_RE.sub("", text)     # remove header fields
    text = _QUOTE_RE.sub("", text)      # remove quoted lines
    text = _URL_RE.sub(" ", text)       # remove URLs
    text = _EMAIL_RE.sub(" ", text)     # remove email addresses
    text = _MULTI_SPACE.sub(" ", text)  # collapse whitespace
    text = text.strip()
    # Truncate to MAX_WORDS words to keep encoding time predictable
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    return text


# ── main pipeline ─────────────────────────────────────────────────────────────

def load_and_clean():
    """Fetch the full 20-newsgroups corpus and return cleaned docs + metadata."""
    log.info("Fetching 20 Newsgroups corpus …")
    # remove_headers/footers/quotes is sklearn's best-effort; we re-clean anyway
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    docs, categories, indices = [], [], []
    skipped = 0
    for idx, (raw, cat_id) in enumerate(zip(data.data, data.target)):
        cleaned = clean(raw)
        # Skip documents that are too short after cleaning – they carry no
        # semantic signal and would distort cluster centroids
        if len(cleaned.split()) < 30:
            skipped += 1
            continue
        docs.append(cleaned)
        categories.append(data.target_names[cat_id])
        indices.append(idx)

    log.info(f"Kept {len(docs):,} documents, skipped {skipped:,} (too short after cleaning)")
    return docs, categories, indices, data.target_names


def embed(docs: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Return L2-normalised embeddings (shape: N × 384)."""
    log.info(f"Loading embedding model '{model_name}' …")
    model = SentenceTransformer(model_name)

    log.info(f"Encoding {len(docs):,} documents in batches of {BATCH_SIZE} …")
    embeddings = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
        convert_to_numpy=True,
    )
    return embeddings  # shape (N, 384)


def ingest_to_chroma(docs, categories, indices, embeddings):
    """Persist documents + embeddings into ChromaDB."""
    log.info(f"Initialising ChromaDB at '{CHROMA_PATH}' …")
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    # Wipe and recreate so this script is idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance index
    )

    log.info("Inserting into ChromaDB …")
    for start in tqdm(range(0, len(docs), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(docs))
        collection.add(
            ids=[str(i) for i in range(start, end)],
            documents=docs[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[
                {"category": categories[i], "orig_index": int(indices[i])}
                for i in range(start, end)
            ],
        )

    log.info(f"ChromaDB collection '{COLLECTION_NAME}' has {collection.count():,} items")
    return collection


def main():
    docs, categories, indices, target_names = load_and_clean()

    embeddings = embed(docs)

    # Persist embeddings for reuse in Part 2 (avoid re-encoding)
    np.save("embeddings.npy", embeddings)
    np.save("doc_indices.npy", np.array(indices))
    import json
    with open("categories.json", "w") as f:
        json.dump(categories, f)
    with open("docs_cleaned.json", "w") as f:
        json.dump(docs, f)

    log.info("Saved embeddings.npy, doc_indices.npy, categories.json, docs_cleaned.json")

    ingest_to_chroma(docs, categories, indices, embeddings)
    log.info("Part 1 complete ✓")


if __name__ == "__main__":
    main()