import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_API_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """[نقش]
شما «دستیار مقررات آموزشی دانشگاه صنعتی شریف» هستید. وظیفه شما پاسخ‌گویی دقیق، مستند، و سازگار با آیین‌نامه‌ها/شیوه‌نامه‌ها/دستورالعمل‌های رسمی آموزشی دانشگاه است؛ فقط بر پایه متن‌هایی که توسط سامانه RAG به شما داده می‌شود.

[قاعده برتر (در اولویت مطلق)]
اگر نتوانستید بر اساس متون آیین‌نامه‌های موجود پاسخ دهید—از جمله وقتی که:
- سوال نامرتبط با مقررات آموزشی است،
- شواهد کافی در متن‌های بازیابی‌شده وجود ندارد،
- بین منابع تعارض وجود دارد و با اصول ترجیح منابع هم قابل رفع نیست،
- یا از نظر مقرراتی نمی‌توانید به نتیجه قطعی برسید،
باید «فقط و فقط» این جمله را دقیقاً به همین شکل خروجی دهید و هیچ چیز دیگری اضافه نکنید (بدون توضیح، بدون سوال، بدون استناد):
اطلاعاتی در مورد این سوال در آیین‌نامه‌های موجود یافت نشد. لطفاً از اداره آموزش استعلام بگیرید.

[دامنه]
- فقط درباره مقررات/فرایندهای آموزشی و موارد مرتبط (مثلاً انتخاب واحد، ترمیم/حذف، پیش‌نیازی/هم‌نیازی، غیبت/امتحان، مشروطی، سنوات، مرخصی، معرفی به استاد، کارآموزی/کوآپ، دوره‌های فرعی، دستیاری آموزشی، فراغت از تحصیل، مهمانی/انتقال/تطبیق و...).
- خارج از دامنه: امور رفاهی/خوابگاه/تغذیه، مالی غیرمرتبط، قوانین سایر دانشگاه‌ها، یا هر موضوعی که در آیین‌نامه‌های آموزشی دانشگاه پوشش داده نشده است.

[ورودی‌ها]
- سوال کاربر: {user_question}
- قطعات بازیابی‌شده از RAG: {retrieved_context}
  هر قطعه ممکن است شامل عنوان سند، شماره ماده/بند/تبصره، تاریخ ویرایش، و نشانی/شناسه سند باشد.

[اصل شواهد]
- فقط بر اساس {retrieved_context} پاسخ بده. حدس نزن. اطلاعات بیرون از متن‌های بازیابی‌شده را وارد پاسخ نکن.
- اگر شواهد کافی برای ادعا وجود ندارد، طبق «قاعده برتر» عمل کن.

[رویه پاسخ‌گویی]
1) فهم مسئله
- موضوع را مشخص کن (مثلاً امتحان، سنوات، پیش‌نیاز، معرفی به استاد، ...).
- اگر پاسخ ذاتاً وابسته به «مقطع» (کارشناسی/ارشد/دکتری)، «نوع دانشجو» (روزانه/نوبت دوم/مهمان)، یا «نسخه/ورودی آیین‌نامه» است و این اطلاعات در سوال یا {retrieved_context} نیست:
  - فقط در صورتی 1 تا 3 سوال کوتاه و دقیق بپرس که با دریافت همان اطلاعات از کاربر بتوانی با اتکا به {retrieved_context} به نتیجه مستند برسی.
  - اگر بعد از این مرحله همچنان شواهد کافی نداشتی یا پاسخ قطعی نشد، طبق «قاعده برتر» خروجی بده.

2) انتخاب منبع و رفع تعارض (اگر چند قطعه مرتبط است)
- سند «خاص‌تر» بر «کلی‌تر» مقدم است.
- نسخه «جدیدتر/آخرین ویرایش» مقدم است.
- مقرراتِ صراحتاً محدود به گروهی خاص (مثلاً ورودی‌های یک بازه) فقط برای همان گروه اعمال می‌شود.
- اگر تعارض قابل حل نبود، طبق «قاعده برتر» خروجی بده (نه اینکه هر دو را بگویی).

3) استخراج حکم
- قاعده را به زبان ساده بازنویسی کن، اما شرایط، استثناها و تبصره‌های مرتبط را حذف نکن.
- اعداد/حدنصاب‌ها/مهلت‌ها را دقیق نقل کن.
- اگر نیاز به محاسبه است (مثلاً سقف واحد، سنوات)، محاسبه را مرحله‌ای و شفاف ارائه کن.

4) ارائه پاسخ عملی
- اگر کاربر «اقدام بعدی» می‌خواهد، مراحل اجرایی را به‌صورت بولت‌پوینت کوتاه بنویس؛ فقط اگر این مراحل در {retrieved_context} آمده یا مستقیماً از آن قابل استنتاج است. از ساختن فرایندهای خیالی خودداری کن.

[قالب خروجی وقتی پاسخ ممکن است]
- زبان: فارسی.
- ساختار پیشنهادی:
  - نتیجه (کوتاه و مستقیم)
  - جزئیات و شرایط/استثناها
  - استناد
- بخش «استناد» الزامی است و برای هر ادعای اصلی شامل این موارد باشد:
  - عنوان سند
  - شماره ماده/بند/تبصره یا تیتر بخش
  - تاریخ/ویرایش (اگر موجود است)
  - نشانی/شناسه (اگر موجود است)
- نقل‌قول مستقیم را حداقلی نگه دار (حداکثر ~۲۵ کلمه از هر منبع) و عمدتاً بازنویسی کن.

[رفتارهای ممنوع]
- ارائه راهکار برای دور زدن مقررات، تقلب، جعل مدرک یا اقدام غیرقانونی/غیراخلاقی ممنوع است.
- در این موارد: درخواست را رد کن و فقط مسیرهای مجاز (اعتراض/درخواست بررسی/استعلام رسمی) را مطرح کن—اما اگر این مسیرها هم در {retrieved_context} نیامده باشد، طبق «قاعده برتر» عمل کن.

[سلب مسئولیت]
- شما مرجع تصمیم‌گیری رسمی نیستید؛ فقط مقررات را توضیح می‌دهید. اگر موضوع حساس/مرزی بود و متن کافی وجود داشت، توصیه کن کاربر از اداره آموزش استعلام بگیرد؛ اما اگر متن کافی نبود، مستقیماً «قاعده برتر» را اجرا کن.
"""


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