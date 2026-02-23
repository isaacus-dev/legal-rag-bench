"""
Build and persist FAISS vector databases for the Legal RAG Bench evaluation.

What this script does:
- Downloads the `isaacus/legal-rag-bench` dataset (corpus config).
- Extracts the text field + per-row metadata.
- For each embedding model in `embedding_models`, embeds the dataset in batches.
- Builds a FAISS index (cosine-style via L2-normalized vectors) and saves it to disk.

To extend support to more embedding models, simply initialize new models via LangChain in `config.py`,
and then add it to the `embedding_models` list.
"""

import os
from math import ceil

from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from config import DB_DIR, embedding_models

# Ensure the base directory for vector DBs exists.
os.makedirs(DB_DIR, exist_ok=True)

# Number of documents to embed/index per iteration.
# Tune this based on memory constraints and embedding throughput.
BATCH_SIZE = 256

# Dataset column containing the document text to embed.
TEXT_FIELD = "text"


def load_data() -> tuple[list[str], list[dict]]:
    """
    Download and preprocess the evaluation dataset.

    Returns:
        texts: List of raw document strings (to be embedded).
        metas: List of metadata dicts aligned with `texts` (same length).
              Each dict contains all dataset fields except the text field.
    """
    print("Downloading dataset...")

    ds = load_dataset(
        "isaacus/legal-rag-bench",
        "corpus",
        split="test",
    )

    texts: list[str] = []
    metas: list[dict] = []

    for row in tqdm(ds, total=len(ds), desc="Processing dataset"):
        text = row[TEXT_FIELD]
        metadata = {k: v for k, v in row.items() if k != TEXT_FIELD}
        texts.append(text)
        metas.append(metadata)

    return texts, metas


def create_db(model_cfg: dict, texts: list[str], metas: list[dict]) -> None:
    """
    Create and persist a FAISS vector store for a single embedding model.

    Args:
        model_cfg: A dict containing:
            - "model_name": short name used for output directory
            - "model": LangChain embeddings object
        texts: Document strings to embed.
        metas: Metadata dicts aligned with `texts`.

    Raises:
        RuntimeError: If there is no data to index.
    """
    model_name = model_cfg["model_name"]
    embedding_model = model_cfg["model"]

    # Each embedding model gets its own persisted FAISS index directory.
    persist_dir = os.path.join(DB_DIR, model_name)
    os.makedirs(persist_dir, exist_ok=True)

    print(f"\nBuilding FAISS (cosine-style) index for model '{model_name}'...")

    db: FAISS | None = None
    n = len(texts)
    num_batches = ceil(n / BATCH_SIZE)

    # We embed and index in batches to reduce peak memory usage and allow
    # incremental construction of the FAISS index.
    for i in tqdm(
        range(0, n, BATCH_SIZE),
        total=num_batches,
        desc=f"Embedding & indexing ({model_name})",
    ):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metas = metas[i : i + BATCH_SIZE]

        if db is None:
            # First batch: create the FAISS store from scratch.
            # normalize_L2=True makes inner-product search behave like cosine
            # similarity (assuming vectors are normalized).
            db = FAISS.from_texts(
                texts=batch_texts,
                embedding=embedding_model,
                metadatas=batch_metas,
                normalize_L2=True,
            )
        else:
            # Subsequent batches: extend the existing index.
            db.add_texts(batch_texts, metadatas=batch_metas)

    if db is None:
        raise RuntimeError("No data to index; db was never created.")

    # Save the FAISS index + metadata locally for later retrieval experiments.
    db.save_local(persist_dir)

    print(f"FAISS vector DB created for model: {model_name}")
    print(f"Persisted to: {persist_dir}")


if __name__ == "__main__":
    # Step 1: Load documents once.
    texts, metas = load_data()

    # Step 2: Build one FAISS DB per embedding model configuration.
    for model_cfg in embedding_models:
        create_db(model_cfg, texts, metas)
