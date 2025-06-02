import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from olmocr.data.renderpdf import render_pdf_to_base64png

_cached_model = None


def run_moondream(
    pdf_path: str,
    page_num: int = 1,
    model: str = "vikhyatk/moondream-next",
) -> str:
    global _cached_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _cached_model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        model = model.to(device)
        model.compile()

        _cached_model = model
    else:
        model = _cached_model

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num)
    pil_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    return model.query(
        pil_image,
        "Transcribe the text in natural reading order.",
        settings={"temperature": 0.0, "max_tokens": 1200},
    )["answer"]
