"""
Instruction templates cho InstructABSA — ATSC subtask (given-aspect).

Bám sát convention paper gốc (Scaria et al. 2024, NAACL):
  - Set-2: definition + 2 positive + 2 negative + 2 neutral examples
  - Cấu trúc:
      <bos_instruct>
      Now complete the following example -
      Input: <sentence> | <aspect>
      Output:

Tách 2 ngôn ngữ: EN dùng cho tk-instruct-base-def-pos (đã instruct-tune
tiếng Anh), VI dùng cho VietAI/vit5-base (T5 pretrain trên 138GB text
Việt thuần, không instruct-tune nên prompt phải bằng tiếng Việt thuần
để khớp pretrain).

Few-shot examples cho EN lấy từ SemEval-14 (Rest + Laptop) — in-domain;
VI examples self-construct theo phong cách UIT-VSFC.
"""
from __future__ import annotations

from typing import Literal

Language = Literal["en", "vi"]


# ──────────────────────────── ENGLISH (ATSC) ────────────────────────────

EN_DEFINITION = (
    "Definition: The output will be the sentiment polarity of the given "
    "aspect with respect to the review. Choose exactly one of: positive, "
    "negative, neutral."
)

EN_POS_EXAMPLES = [
    ("The bread is top notch as well. | food",                      "positive"),
    ("Great laptop that offers many great features! | features",    "positive"),
]

EN_NEG_EXAMPLES = [
    ("But the staff was so horrible to us. | service",              "negative"),
    ("The screen is too small and the keyboard sticks. | screen",   "negative"),
]

EN_NEU_EXAMPLES = [
    ("It's a typical Italian restaurant in midtown. | anecdotes/miscellaneous", "neutral"),
    ("The cord is detachable. | cord",                              "neutral"),
]

EN_DELIM   = "Now complete the following example -"
EN_EOS_FMT = "Output:"


# ──────────────────────────── VIETNAMESE (ATSC) ────────────────────────────

VI_DEFINITION = (
    "Định nghĩa: Đầu ra là cực cảm xúc của khía cạnh được cho trong đánh "
    "giá. Chọn đúng một trong: positive, negative, neutral."
)

VI_POS_EXAMPLES = [
    ("nói tiếng anh lưu loát . | lecturer",                         "positive"),
    ("slide giáo trình đầy đủ . | training_program",                "positive"),
]

VI_NEG_EXAMPLES = [
    ("giáo viên không giảng dạy kiến thức . | lecturer",            "negative"),
    ("phòng học rất nóng và thiếu máy chiếu . | facility",          "negative"),
]

VI_NEU_EXAMPLES = [
    ("học phần kéo dài 15 tuần . | training_program",               "neutral"),
    ("trường có căng tin ở tầng trệt . | facility",                 "neutral"),
]

VI_DELIM   = "Bây giờ hoàn thành ví dụ sau -"
VI_EOS_FMT = "Output:"


# ──────────────────────────── BUILDERS ────────────────────────────

def _build_bos_instruct(definition: str,
                        pos: list[tuple[str, str]],
                        neg: list[tuple[str, str]],
                        neu: list[tuple[str, str]]) -> str:
    """Build the bos_instruct (definition + 2 pos + 2 neg + 2 neut) — Set-2."""
    parts: list[str] = [definition, ""]
    for i, (inp, out) in enumerate(pos, 1):
        parts.append(f"Positive Example {i} -")
        parts.append(f"Input: {inp}")
        parts.append(f"Output: {out}")
        parts.append("")
    for i, (inp, out) in enumerate(neg, 1):
        parts.append(f"Negative Example {i} -")
        parts.append(f"Input: {inp}")
        parts.append(f"Output: {out}")
        parts.append("")
    for i, (inp, out) in enumerate(neu, 1):
        parts.append(f"Neutral Example {i} -")
        parts.append(f"Input: {inp}")
        parts.append(f"Output: {out}")
        parts.append("")
    return "\n".join(parts).rstrip()


EN_BOS = _build_bos_instruct(EN_DEFINITION, EN_POS_EXAMPLES, EN_NEG_EXAMPLES, EN_NEU_EXAMPLES)
VI_BOS = _build_bos_instruct(VI_DEFINITION, VI_POS_EXAMPLES, VI_NEG_EXAMPLES, VI_NEU_EXAMPLES)


def build_prompt(input_text: str, language: Language = "en") -> str:
    """Compose full prompt = bos_instruct + delim + Input/Output skeleton.

    `input_text` là chuỗi đã có sẵn dạng "<sentence> | <aspect>"
    (xem scripts/prepare_instruct_absa.py).
    """
    if language == "en":
        bos, delim, eos = EN_BOS, EN_DELIM, EN_EOS_FMT
    elif language == "vi":
        bos, delim, eos = VI_BOS, VI_DELIM, VI_EOS_FMT
    else:
        raise ValueError(f"Unsupported language: {language!r}")
    return f"{bos}\n\n{delim}\nInput: {input_text}\n{eos}"


def get_bos_instruct(language: Language = "en") -> str:
    """Trả về bos_instruct đã build sẵn (dùng cho debug/inspect)."""
    return EN_BOS if language == "en" else VI_BOS


if __name__ == "__main__":
    print("=== EN prompt sample ===")
    print(build_prompt("The food was great but service was slow. | service", "en"))
    print()
    print("=== VI prompt sample ===")
    print(build_prompt("giảng bài rất nhiệt tình . | lecturer", "vi"))
