# ABSA-SOTA-Survey: Khảo sát Trade-off của 5 Paradigm trên SemEval-2014 & UIT-VSFC

Repository thực nghiệm 5 phương pháp ABSA SOTA (2019–2025) trên tiếng Anh (SemEval-2014)
và tiếng Việt (UIT-VSFC), tập trung phân tích trade-off **Accuracy vs Latency** và khả
năng generalize đa ngôn ngữ.

---

## Mục lục

1. [Phương pháp khảo sát](#1-phương-pháp-khảo-sát)
2. [Dataset](#2-dataset)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt](#4-cài-đặt)
5. [Chuẩn bị dữ liệu](#5-chuẩn-bị-dữ-liệu)
6. [Giao thức đánh giá chung](#6-giao-thức-đánh-giá-chung)
7. [Hướng dẫn từng model](#7-hướng-dẫn-từng-model)
8. [Tổng hợp kết quả & biểu đồ](#8-tổng-hợp-kết-quả--biểu-đồ)

---

## 1. Phương pháp khảo sát

| # | Method | Year | Paradigm | Backbone EN | Backbone VI |
|---|--------|------|----------|-------------|-------------|
| 1 | LCF-BERT | 2019 | Discriminative | bert-base-uncased | vinai/phobert-base |
| 2 | InstructABSA | 2023 | Instruction-Tuning | allenai/tk-instruct-base-def-pos | google/mt5-base |
| 3 | SSIN | 2024 | Graph (Syn+Sem) | bert-base-uncased + spaCy | phobert-base + VnCoreNLP |
| 4 | DOT | 2025 | Generative-Seq2Seq | t5-base | VietAI/vit5-base |
| 5 | LLM-Reasoning | 2025 | LLM-Reasoning | gpt-4o / llama3 | SeaLLMs/SeaLLM-7B-v2 |

---

## 2. Dataset

### SemEval-2014 (English)
- **Task**: ABSA — predict aspect category + sentiment
- **Domain**: Restaurant reviews (+ Laptop)
- **Nhãn aspect**: `food` / `service` / `price` / `ambience` / `anecdotes/miscellaneous`
- **Nhãn sentiment**: `positive` / `negative` / `neutral` / `conflict`
- **File raw**: `data/raw/semeval14/*.xml`

### UIT-VSFC (Vietnamese)
- **Task**: ABSA — predict topic + sentiment (single-label per sentence)
- **Domain**: Student feedback (giáo dục đại học)
- **Nhãn topic**: `lecturer` / `training_program` / `facility` / `others`
- **Nhãn sentiment**: `positive` / `negative` / `neutral`
- **File raw**: `data/raw/uit-vsfc/{train,val,test}.csv`

---

## 3. Cấu trúc thư mục

```
absa-sota-survey/
├── README.md
├── requirements.txt
├── evaluate.py                  # runner đánh giá chung (KHÔNG sửa)
│
├── eval/                        # metrics engine
│   ├── schema.py                # constants + validate_predictions_file()
│   └── score.py                 # accuracy + macro-F1, write_metrics()
│
├── data/
│   ├── raw/
│   │   ├── semeval14/           # Restaurants_Train_v2.xml, Laptops_Test_Gold.xml, ...
│   │   └── uit-vsfc/            # train.csv, val.csv, test.csv
│   └── processed/
│       ├── lcf_bert/            # JSONL cho LCF-BERT (đã có)
│       ├── instruct_absa/       # (mỗi người tự tạo cho model mình)
│       ├── ssin/
│       ├── dot/
│       └── llm_reasoning/
│
├── models/
│   └── lcf_bert/
│       ├── lcf_bert.py          # kiến trúc dual-head
│       ├── data_utils.py        # ABSADatasetJSONL + Tokenizer4Bert
│       └── train.py             # training script
│   ├── instruct_absa/
│   ├── ssin/
│   ├── dot/
│   └── llm_reasoning/
│
├── predictors/                  # MỖI NGƯỜI CODE MỘT FILE Ở ĐÂY
│   ├── lcf_bert.py              # 
│   ├── instruct_absa.py         # stub -> bạn phụ trách InstructABSA điền vào
│   ├── ssin.py                  # stub -> bạn phụ trách SSIN điền vào
│   ├── dot.py                   # stub -> bạn phụ trách DOT điền vào
│   └── llm_reasoning.py         # stub -> OpenAI/Ollama (đã có một phần)
│
├── configs/
│   ├── lcf_bert_en.yaml         # SemEval-2014-Restaurant
│   └── lcf_bert_vi.yaml         # UIT-VSFC (PhoBERT)
│
└── scripts/
    ├── prepare_semeval14_lcf_bert.py     # XML → JSONL 
    ├── prepare_vsfc_lcf_bert.py          # CSV → JSONL
    └── plot_tradeoff.py         # vẽ biểu đồ accuracy vs latency
```

---

## 4. Cài đặt

```bash
git clone <repo-url>
cd absa-sota-survey
pip install -r requirements.txt
```

---

## 5. Chuẩn bị dữ liệu

Chỉ cần chạy một lần cho LCF-BERT (output đã có trong `data/processed/lcf_bert/`):

```bash
python scripts/prepare_semeval14_lcf_bert.py   # -> data/processed/lcf_bert/semeval14_*.jsonl
python scripts/prepare_vsfc_lcf_bert.py        # -> data/processed/lcf_bert/vsfc_*.jsonl
```

**Các model khác** có thể cần format riêng — lưu vào `data/processed/<tên_model>/`.

---

## 6. Giao thức đánh giá chung

> Đây là phần **quan trọng nhất** — tất cả 5 người phải làm theo đúng để kết quả
> có thể so sánh được.

### 6.1 Quy tắc output thống nhất

Mỗi model **phải** trả về cùng 3 giá trị:

```python
predict(text: str, aspect=None) -> (pred_aspect: str, pred_sentiment: str, raw_output: str)
```

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `pred_aspect` | `str` | nhãn aspect/topic dự đoán (xem bảng nhãn §2) |
| `pred_sentiment` | `str` | `"positive"` / `"negative"` / `"neutral"` |
| `raw_output` | `str` | output thô của model (để debug) |

Nếu model không parse được output → trả về `("__PARSE_ERROR__", "__PARSE_ERROR__", raw)`.

### 6.2 Metrics được tính tự động

`evaluate.py` + `eval/score.py` tính các chỉ số sau cho mỗi lần chạy:

| Metric | Ý nghĩa |
|--------|---------|
| `sentiment_accuracy` | % câu dự đoán đúng sentiment |
| `sentiment_macro_f1` | macro-F1 trên tất cả lớp sentiment |
| `aspect_accuracy` | % câu dự đoán đúng aspect/topic |
| `aspect_macro_f1` | macro-F1 trên tất cả lớp aspect |
| `joint_accuracy` | % câu đúng **cả hai** (aspect + sentiment) |
| `avg_latency_ms` | thời gian trung bình mỗi câu (ms) |
| `parse_error_rate` | tỉ lệ output không parse được |

### 6.3 Cách implement Predictor cho model của bạn

**Bước 1** — Tạo file `predictors/<tên_model>.py` (stub đã có sẵn).

**Bước 2** — Implement đúng class với 3 thuộc tính + 1 method bắt buộc:

```python
class MyModelPredictor:
    # ── 3 thuộc tính bắt buộc ─────────────────────────────────────────────
    method   = "TênModelCủaBạn"          # xuất hiện trong bảng kết quả
    paradigm = "Discriminative"          # xem bảng §1
    backbone = "bert-base-uncased"       # model HuggingFace sử dụng

    def __init__(self, checkpoint: str, device: str = "cuda"):
        # load model, tokenizer, config...
        pass

    # ── Method bắt buộc ───────────────────────────────────────────────────
    def predict(self, text: str, aspect=None) -> tuple[str, str, str]:
        """
        Nhận vào: text (câu tiếng Anh hoặc tiếng Việt)
        Trả về:   (pred_aspect, pred_sentiment, raw_output)

        KHÔNG được nhận aspect làm input — model phải tự predict cả hai.
        """
        # ... inference ...
        return pred_aspect, pred_sentiment, raw_output

    # ── Optional: warmup (tránh đo thời gian JIT) ─────────────────────────
    def warmup(self, text: str) -> None:
        self.predict(text)
```

**Bước 3** — Chạy đánh giá:

```bash
# Đánh giá trên SemEval-2014-Restaurant
python evaluate.py \
    --predictor predictors.<tên_module>:<NhãnClass> \
    --predictor-kwargs '{"checkpoint": "checkpoints/<tên_model>/best.pt"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
    --output-dir results

# Đánh giá trên UIT-VSFC
python evaluate.py \
    --predictor predictors.<tên_module>:<NhãnClass> \
    --predictor-kwargs '{"checkpoint": "checkpoints/<tên_model>/best.pt"}' \
    --test-set data/processed/lcf_bert/vsfc_test.jsonl \
    --output-dir results
```

Output gồm:
- `results/predictions/<method>_<dataset>.jsonl` — toàn bộ dự đoán
- `results/metrics/<method>_<dataset>.json` — tất cả metrics + latency

### 6.4 Ví dụ chạy thực tế (LCF-BERT)

```bash
# Bước 1: Huấn luyện
python models/lcf_bert/train.py --config configs/lcf_bert_en.yaml

# Bước 2: Đánh giá
python evaluate.py \
    --predictor predictors.lcf_bert:LCFBertPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/semeval14_rest/lcf_bert_best.pt"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
    --output-dir results
```

### 6.5 Nhãn hợp lệ

Khi predict, hãy đảm bảo output **đúng chính xác** các chuỗi sau (lowercase):

**Sentiment** (cả 2 dataset):
```
positive   negative   neutral
```

**Aspect — SemEval-2014-Restaurant**:
```
food   service   price   ambience   anecdotes/miscellaneous
```

**Aspect — SemEval-2014-Laptop** (aspect term, open vocabulary — ví dụ):
```
battery life   screen   keyboard   price   ...
```

**Aspect — UIT-VSFC**:
```
lecturer   training_program   facility   others
```

---

## 7. Hướng dẫn từng model

### LCF-BERT (đã có)
```bash
python models/lcf_bert/train.py --config configs/lcf_bert_en.yaml   # SemEval14
python models/lcf_bert/train.py --config configs/lcf_bert_vi.yaml   # UIT-VSFC
```
Predictor: `predictors/lcf_bert.py` — đã implement đầy đủ.

### InstructABSA
- File cần điền: `predictors/instruct_absa.py`
- Load HuggingFace checkpoint, format prompt, gọi generate, parse output → `(aspect, sentiment)`
- Tham khảo paper: [arXiv 2302.05001](https://arxiv.org/abs/2302.05001)

### SSIN
- File cần điền: `predictors/ssin.py`
- Cần dependency: spaCy (EN) / VnCoreNLP (VI) để tạo dependency tree
- Predictor: `predictors/ssin.py`

### DOT
- File cần điền: `predictors/dot.py`
- T5-based seq2seq — format input prompt → generate → parse "aspect: X, sentiment: Y"
- Backbone: `t5-base` (EN) / `VietAI/vit5-base` (VI)

### LLM-Reasoning
- File đã có một phần: `predictors/llm_reasoning.py`
- Hỗ trợ sẵn OpenAI API và Ollama local
- Zero-shot với structured JSON prompt — không cần fine-tune
```bash
# Dùng Ollama local
python evaluate.py \
    --predictor predictors.llm_reasoning:LLMReasoningPredictor \
    --predictor-kwargs '{"backend":"ollama","model":"llama3"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl
```

---

## 8. Tổng hợp kết quả & biểu đồ

Sau khi **tất cả 5 model** đã chạy `evaluate.py`, kết quả nằm ở `results/metrics/*.json`.

```bash
# Vẽ biểu đồ trade-off Latency vs Accuracy
python scripts/plot_tradeoff.py --out results/tradeoff.png

# Xem nhanh tất cả metrics
python -c "
import json, glob
for p in sorted(glob.glob('results/metrics/*.json')):
    m = json.load(open(p))
    print(f\"{m['method']:20s} {m['dataset']:30s}  \
sent_acc={m['sentiment_accuracy']:.3f}  \
macro_f1={m['sentiment_macro_f1']:.3f}  \
latency={m.get('efficiency',{}).get('avg_latency_ms','N/A')}ms\")
"
```

