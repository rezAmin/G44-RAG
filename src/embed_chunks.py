import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CHUNKS_PATH = os.path.join(DATA_DIR, "sharif_rules_chunks.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(DATA_DIR, "chunk_mapping.json")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_texts(chunks: list[dict]) -> list[str]:
    """
    multilingual-e5 expects 'passage: ...' prefix for documents
    and 'query: ...' prefix for queries at search time.
    """
    texts = []
    for chunk in chunks:
        header = f"{chunk['rule_title']} — {chunk['section_title']}"
        text = f"passage: {header}\n{chunk['content']}"
        texts.append(text)
    return texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_mapping(chunks: list[dict], path: str):
    mapping = []
    for i, chunk in enumerate(chunks):
        mapping.append({
            "index": i,
            "id": chunk["id"],
            "rule_title": chunk["rule_title"],
            "rule_url": chunk["rule_url"],
            "rule_date": chunk["rule_date"],
            "parent_section": chunk["parent_section"],
            "section_title": chunk["section_title"],
            "content": chunk["content"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Preparing texts...")
    texts = prepare_texts(chunks)

    print("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    embeddings = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving chunk mapping to {MAPPING_PATH}")
    save_mapping(chunks, MAPPING_PATH)

    print("Done! Index and mapping saved.")


if __name__ == "__main__":
    main()
