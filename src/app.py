import os

import gradio as gr
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()

pipeline = None


def initialize():
    global pipeline
    if pipeline is None:
        print("Initializing RAG pipeline...")
        pipeline = RAGPipeline(top_k=5)
        print("Pipeline ready.")


def chat(query: str, history: list) -> str:
    if not query.strip():
        return "لطفاً سوال خود را وارد کنید."

    result = pipeline.answer(query)
    answer = result["answer"]

    sources_text = "\n\n---\n**منابع:**\n"
    for s in result["sources"]:
        sources_text += f"- {s['rule_title']} — {s['section_title']} (امتیاز: {s['score']:.3f})\n"

    return answer + sources_text


def build_ui():
    mode = os.getenv("GENERATOR_MODE", "local").strip().lower()
    model_label = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct") if mode == "api" else "Qwen2.5-7B-Instruct (local)"
    mode_badge = f"🌐 API mode — `{model_label}`" if mode == "api" else f"💻 Local mode — `{model_label}`"

    with gr.Blocks(
        title="چت‌بات مقررات دانشگاه شریف",
        theme=gr.themes.Soft(),
        css="footer {display: none !important}",
    ) as demo:
        gr.Markdown(
            f"""
            # چت‌بات راهنمای مقررات آموزشی دانشگاه صنعتی شریف
            سوالات خود درباره آیین‌نامه‌ها و مقررات آموزشی را بپرسید.
            پاسخ‌ها فقط بر اساس اسناد رسمی دانشگاه ارائه می‌شود.

            **Generator:** {mode_badge}
            """
        )

        chatbot = gr.ChatInterface(
            fn=chat,
            examples=[
                "شرایط مشروطی دانشجو چیست؟",
                "حداکثر سنوات مجاز تحصیل در مقطع کارشناسی چقدر است؟",
                "شرایط حذف اضطراری درس چیست؟",
                "آیا استفاده از هوش مصنوعی در تکالیف مجاز است؟",
                "شرایط مهمانی در دانشگاه دیگر چیست؟",
                "قوانین کارآموزی چیست؟",
            ],
        )

    return demo


if __name__ == "__main__":
    initialize()
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
