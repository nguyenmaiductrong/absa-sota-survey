# ABSA-SOTA-Survey: Khảo sát Trade-off của 5 Paradigm trên SemEval-2014 & UIT-VSFC

Repository thực nghiệm 5 phương pháp ABSA SOTA (2019–2025) trên tiếng Anh (SemEval-2014) và tiếng Việt (UIT-VSFC), tập trung phân tích trade-off **Accuracy vs Latency** và khả năng generalize đa ngôn ngữ.

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
- **Nhãn aspect (Restaurant)**: `food` / `service` / `price` / `ambience` / `anecdotes/miscellaneous`
- **Nhãn aspect (Laptop)**: open vocabulary (aspect term)
- **Nhãn sentiment**: `positive` / `negative` / `neutral` (3 lớp; nhãn `conflict` bị loại)
- **File raw**: `data/raw/semeval14/*.xml`

### UIT-VSFC (Vietnamese)
- **Task**: ABSA — predict topic + sentiment (single-label per sentence)
- **Nhãn topic**: `lecturer` / `training_program` / `facility` / `others`
- **Nhãn sentiment**: `positive` / `negative` / `neutral`
- **File raw**: `data/raw/uit-vsfc/{train,val,test}.csv`

---

## 3. Metrics so sánh chung

| Metric | Ý nghĩa |
|--------|---------|
| `sentiment_accuracy` | % câu dự đoán đúng sentiment |
| `sentiment_macro_f1` | Macro-F1 trên tất cả lớp sentiment |
| `avg_latency_ms` | Thời gian trung bình mỗi câu (ms) |

---

## 4. Giao thức đánh giá chung

Mỗi model **phải** implement class với 3 thuộc tính + 1 method:

```python
class MyModelPredictor:
    method   = "TênModel"
    paradigm = "Discriminative"
    backbone = "bert-base-uncased"

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        # Trả về (pred_aspect, pred_sentiment, raw_output)
        # Nếu không parse được: return ("__PARSE_ERROR__", "__PARSE_ERROR__", raw)
        ...
```

Chạy đánh giá:

```bash
python evaluate.py \
    --predictor predictors.<module>:<Class> \
    --predictor-kwargs '{"checkpoint": "checkpoints/<model>/best.pt"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
    --output-dir results
```

Output:
- `results/predictions/<method>_<dataset>.jsonl`
- `results/metrics/<method>_<dataset>.json`

---

## 5. LCF-BERT (phần này tôi phụ trách)

### Cài đặt

```bash
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu (chạy một lần)

```bash
python scripts/prepare_semeval14_lcf_bert.py   # -> data/processed/lcf_bert/semeval14_*.jsonl
python scripts/prepare_vsfc_lcf_bert.py        # -> data/processed/lcf_bert/vsfc_*.jsonl
```

### Train

```bash
# SemEval-2014 Restaurant (EN)
python models/lcf_bert/train.py --config configs/lcf_bert_en_restaurant.yaml

# SemEval-2014 Laptop (EN)
python models/lcf_bert/train.py --config configs/lcf_bert_en_laptop.yaml

# UIT-VSFC (VI)
python models/lcf_bert/train.py --config configs/lcf_bert_vi.yaml
```

Checkpoint lưu tại `checkpoints/semeval14_rest/lcf_bert_best.pt`, `checkpoints/semeval14_lap/lcf_bert_best.pt`, `checkpoints/vsfc/lcf_bert_best.pt`.

### Evaluate

```bash
# SemEval-2014 Restaurant
python evaluate.py \
    --predictor predictors.lcf_bert:LCFBertPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/semeval14_rest/lcf_bert_best.pt"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
    --given-aspect \
    --output-dir results

# SemEval-2014 Laptop
python evaluate.py \
    --predictor predictors.lcf_bert:LCFBertPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/semeval14_lap/lcf_bert_best.pt"}' \
    --test-set data/processed/lcf_bert/semeval14_lap_test.jsonl \
    --given-aspect \
    --output-dir results

# UIT-VSFC
python evaluate.py \
    --predictor predictors.lcf_bert:LCFBertPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/vsfc/lcf_bert_best.pt"}' \
    --test-set data/processed/lcf_bert/vsfc_test.jsonl \
    --given-aspect \
    --output-dir results
```

> `--given-aspect`: truyền gold aspect làm LCF anchor, model chỉ predict sentiment (chế độ ABSC chuẩn của LCF-BERT).

---

## 6. Tổng hợp kết quả

Sau khi tất cả 5 model chạy `evaluate.py`, kết quả nằm ở `results/metrics/*.json`.

```bash
# Xem nhanh bảng so sánh
python -c "
import json, glob
for p in sorted(glob.glob('results/metrics/*.json')):
    m = json.load(open(p))
    print(f\"{m['method']:20s} {m['dataset']:30s}  sent_acc={m['sentiment_accuracy']:.3f}  macro_f1={m['sentiment_macro_f1']:.3f}  latency={m['avg_latency_ms']}ms\")
"

# Vẽ biểu đồ trade-off Latency vs Accuracy
python scripts/plot_tradeoff.py --out results/tradeoff.png
```
