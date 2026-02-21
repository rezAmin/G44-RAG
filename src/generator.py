import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_API_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """تو یک دستیار هوشمند دانشگاهی هستی که فقط بر اساس آیین‌نامه‌ها و مقررات رسمی دانشگاه صنعتی شریف پاسخ می‌دهی.

قوانین پاسخ‌دهی:
۱. فقط بر اساس متن مقررات ارائه‌شده پاسخ بده. هرگز اطلاعاتی خارج از این متون اضافه نکن.
۲. در پاسخ، حتماً نام آیین‌نامه و در صورت امکان شماره ماده، بند یا تبصره را ذکر کن.
۳. پاسخ را کوتاه، دقیق و مستقیم بنویس.
۴. اگر پاسخ سوال در متون ارائه‌شده وجود ندارد، بگو: «اطلاعاتی در مورد این سوال در آیین‌نامه‌های موجود یافت نشد. لطفاً از اداره آموزش استعلام بگیرید.»
۵. هرگز قانون یا تفسیری از خودت اختراع نکن.
۶. به زبان فارسی و رسمی پاسخ بده."""


def build_prompt(query: str, context: str) -> list[dict]:
    user_message = f"""بر اساس متون آیین‌نامه‌ای زیر، به سوال دانشجو پاسخ بده.

--- متون مرتبط ---
{context}
--- پایان متون ---

سوال: {query}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


class Generator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        quantization: str = "4bit",
        device_map: str = "auto",
    ):
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        messages = build_prompt(query, context)

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

class APIGenerator:

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_API_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        messages = build_prompt(query, context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content.strip()