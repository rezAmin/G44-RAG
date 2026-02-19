import json
import csv
import os
from datetime import datetime
from src.rag_pipeline import RAGPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation")

SAMPLE_QUESTIONS = [
    "شرایط مشروطی دانشجوی کارشناسی چیست؟",
    "حداکثر سنوات مجاز تحصیل در دوره کارشناسی چقدر است؟",
    "آیا استفاده از ابزار هوش مصنوعی در تکالیف درسی مجاز است؟",
    "شرایط حذف اضطراری درس چیست؟",
    "قوانین غیبت در امتحان پایان‌ترم چیست؟",
    "شرایط معرفی به استاد چگونه است؟",
    "قوانین کارآموزی در دوره کارشناسی چیست؟",
    "شرایط مهمانی دانشجو در دانشگاه دیگر چگونه است؟",
    "قوانین تغییر رشته در دوره کارشناسی چیست؟",
    "شرایط پروژه کارشناسی چگونه است؟",
    "آیین‌نامه دوره کوآپ چه مقرراتی دارد؟",
    "شرایط انتقال به دانشگاه صنعتی شریف چیست؟",
    "مهلت فراغت از تحصیل چقدر است؟",
    "حداقل و حداکثر واحد مجاز در هر ترم چقدر است؟",
    "شرایط دستیاری آموزشی چیست؟",
]


def run_evaluation(
    pipeline: RAGPipeline,
    questions: list[str],
    output_dir: str = OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        result = pipeline.answer(question)
        results.append(result)
        print(f"  → {result['answer'][:100]}...")
        print()

    json_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON results saved to {json_path}")

    csv_path = os.path.join(output_dir, f"eval_results_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "answer", "num_sources",
            "source_1_title", "source_1_section", "source_1_score",
        ])
        for r in results:
            top_source = r["sources"][0] if r["sources"] else {}
            writer.writerow([
                r["query"],
                r["answer"],
                r["num_chunks_retrieved"],
                top_source.get("rule_title", ""),
                top_source.get("section_title", ""),
                f"{top_source.get('score', 0):.4f}",
            ])
    print(f"CSV results saved to {csv_path}")

    return results


if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=5)
    run_evaluation(pipeline, SAMPLE_QUESTIONS)
