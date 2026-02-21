import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(DATA_DIR, "chunk_mapping.json")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"


class Retriever:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        mapping_path: str = MAPPING_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.index = faiss.read_index(index_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_text = f"query: {query}"
        query_embedding = self.model.encode(
            [query_text], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            chunk = self.mapping[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank + 1
            results.append(chunk)

        return results


def format_retrieved_context(results: list[dict]) -> str:
    context_parts = []
    for r in results:
        header = f"[{r['rule_title']} | {r['section_title']}]"
        context_parts.append(f"{header}\n{r['content']}")
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    retriever = Retriever()
    test_query = "شرایط مشروطی دانشجو چیست؟"
    print(f"Query: {test_query}\n")

    results = retriever.retrieve(test_query, top_k=3)
    for r in results:
        print(f"[Rank {r['rank']}] Score: {r['score']:.4f}")
        print(f"  Rule: {r['rule_title']}")
        print(f"  Section: {r['section_title']}")
        print(f"  Content: {r['content'][:150]}...")
        print()
