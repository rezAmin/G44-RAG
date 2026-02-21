import os

from dotenv import load_dotenv

from src.retriever import Retriever, format_retrieved_context
from src.generator import Generator, APIGenerator

load_dotenv()


def create_generator() -> Generator | APIGenerator:
    mode = os.getenv("GENERATOR_MODE", "local").strip().lower()

    if mode == "api":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "GENERATOR_MODE=api requires OPENROUTER_API_KEY to be set."
            )
        model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
        return APIGenerator(api_key=api_key, model=model)

    return Generator()


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | APIGenerator | None = None,
        top_k: int = 5,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or create_generator()
        self.top_k = top_k

    def answer(self, query: str) -> dict:
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        context = format_retrieved_context(retrieved)
        answer = self.generator.generate(query, context)

        sources = []
        for r in retrieved:
            sources.append({
                "rule_title": r["rule_title"],
                "section_title": r["section_title"],
                "score": r["score"],
                "content_preview": r["content"][:200],
            })

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_chunks_retrieved": len(retrieved),
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    result = pipeline.answer("شرایط مشروطی دانشجو چیست؟")
    print(f"Q: {result['query']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources ({result['num_chunks_retrieved']}):")
    for s in result["sources"]:
        print(f"  - {s['rule_title']} | {s['section_title']} (score: {s['score']:.4f})")
