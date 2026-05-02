# ABSA-SOTA-Survey: Khảo sát Trade-off của 5 Paradigm trên SemEval-2014 & UIT-VSFC

Repository thực nghiệm 5 phương pháp ABSA SOTA (2019–2025) trên tiếng Anh (SemEval-2014) và tiếng Việt (UIT-VSFC), tập trung phân tích trade-off **Accuracy vs Latency** và khả năng generalize đa ngôn ngữ.

---

## 1. Phương pháp khảo sát

| #   | Method          | Year | Paradigm           | Backbone EN                      | Backbone VI              |
| --- | --------------- | ---- | ------------------ | -------------------------------- | ------------------------ |
| 1   | LCF-BERT        | 2019 | Discriminative     | bert-base-uncased                | vinai/phobert-base       |
| 2   | InstructABSA    | 2023 | Instruction-Tuning | allenai/tk-instruct-base-def-pos | VietAI/vit5-base         |
| 3   | SSIN            | 2024 | Graph (Syn+Sem)    | bert-base-uncased + spaCy        | phobert-base + VnCoreNLP |
| 4   | DOT             | 2025 | Generative-Seq2Seq | t5-base                          | VietAI/vit5-base         |
| 5   | Syn-Chain (LLM) | 2025 | LLM-Reasoning      | Qwen 2.5 14B Instruct            | Qwen 2.5 14B Instruct    |

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

| Metric               | Ý nghĩa                            |
| -------------------- | ---------------------------------- |
| `sentiment_accuracy` | % câu dự đoán đúng sentiment       |
| `sentiment_macro_f1` | Macro-F1 trên tất cả lớp sentiment |
| `avg_latency_ms`     | Thời gian trung bình mỗi câu (ms)  |

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

## 6. InstructABSA

> Paper: "InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis" (Scaria et al., NAACL 2024). Paradigm Instruction-Tuning, sub-task ATSC (given-aspect → predict polarity), Set-2 prompt (definition + 2 pos/neg/neu examples).

### Cài đặt

```bash
pip install -r requirements.txt
pip install "transformers>=4.45,<5.0" accelerate sentencepiece
```

> ViT5 (VI backbone) là T5 family → cần `sentencepiece`.

### Chuẩn bị dữ liệu (chạy một lần)

```bash
python scripts/prepare_instruct_absa.py    # -> data/processed/instruct_absa/{semeval14_rest,semeval14_lap,vsfc}_{train,val,test}.jsonl
```

Tái dùng split của LCF-BERT (cùng test set → so sánh trực tiếp); thêm field `prompt` theo Set-2 instruction template.

### Train

Backbone: EN dùng `allenai/tk-instruct-base-def-pos` (220M, đã instruct-tune Super-NaturalInstructions); VI dùng `VietAI/vit5-base` (220M, T5 pretrain trên 138GB text Việt thuần).

```bash
# SemEval-2014 Restaurant (EN)
python -m models.instruct_absa.train --config configs/instruct_absa_en_restaurant.yaml

# SemEval-2014 Laptop (EN)
python -m models.instruct_absa.train --config configs/instruct_absa_en_laptop.yaml

# UIT-VSFC (VI)
python -m models.instruct_absa.train --config configs/instruct_absa_vi.yaml
```

Cấu hình mặc định: 100% data, fp16, batch 4 + grad accum 4 (eff. bs 16), 4 epochs, T4 16GB. Checkpoint lưu tại `checkpoints/{semeval14_rest,semeval14_lap,vsfc}/instruct_absa_best/`.

### Evaluate

```bash
# SemEval-2014 Restaurant
python evaluate.py \
    --predictor predictors.instruct_absa:InstructABSAPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/semeval14_rest/instruct_absa_best","language":"en"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
    --given-aspect --output-dir results

# SemEval-2014 Laptop
python evaluate.py \
    --predictor predictors.instruct_absa:InstructABSAPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/semeval14_lap/instruct_absa_best","language":"en"}' \
    --test-set data/processed/lcf_bert/semeval14_lap_test.jsonl \
    --given-aspect --output-dir results

# UIT-VSFC
python evaluate.py \
    --predictor predictors.instruct_absa:InstructABSAPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/vsfc/instruct_absa_best","language":"vi"}' \
    --test-set data/processed/lcf_bert/vsfc_test.jsonl \
    --given-aspect --output-dir results
```

> `--given-aspect`: gold aspect được truyền cho predictor; mô hình echo aspect và chỉ generate polarity (`positive`/`negative`/`neutral`).

### Chạy nhanh trên Kaggle T4

Cách thuận tiện nhất: dùng `notebook/02-instruct-absa-train-eval.ipynb` (đã viết sẵn upload bundle → train 3 dataset → evaluate → đóng gói results). Quy trình:

```bash
python scripts/make_colab_bundle.py        # -> instruct_absa_bundle.zip
```

Upload `instruct_absa_bundle.zip` lên Kaggle Dataset, mở notebook, Run All. Output zip xuất hiện ở tab Output → tải về.

### Tổng hợp kết quả

```bash
python scripts/summarize_instruct_absa_results.py    # in bảng acc/F1/latency 3 dataset
```

---

## 7. Syn-Chain (LLM-Reasoning) (phần này tôi phụ trách)

### Cài đặt

Cài đặt các thư viện cần thiết và tải model spaCy cho phân tích cú pháp:

```bash
pip install -r requirements.txt
pip install langchain-openai python-dotenv spacy
python -m spacy download en_core_web_sm
```

### Cấu hình biến môi trường

Tạo file `.env` ở thư mục gốc (hoặc export biến môi trường) để cấu hình LLM Qwen 2.5 14B Instruct:

```env
QWEN_API_BASE=http://localhost:8000/v1
QWEN_API_KEY=EMPTY
MODEL_NAME=qwen2.5:14b-instruct
```

### Đánh giá (Evaluate)

Sử dụng script đánh giá của Syn-Chain để thực hiện quá trình phân tích 3 bước (Cú pháp -> Quan điểm -> Cảm xúc) bằng Qwen 2.5 14B Instruct.

```bash
# SemEval-2014 Laptop (EN)
python models/syn-chain-LLM/evaluate.py \
    --data data/processed/syn-chain/laps_semeval.json \
    --out results/predictions/evaluation_logs_laps_semeval.json

# SemEval-2014 Restaurant (EN)
python models/syn-chain-LLM/evaluate.py \
    --data data/processed/syn-chain/restaurant_semeval.json \
    --out results/predictions/evaluation_logs_restaurant_semeval.json

# UIT-VSFC (VI)
python models/syn-chain-LLM/evaluate.py \
    --data data/processed/syn-chain/uit_vsfc.json \
    --out results/predictions/evaluation_logs_uit_vsfc.json
```

Logs kết quả chi tiết kèm lý luận (LLM reasoning) sẽ được lưu tại thư mục `results/predictions/`.

## 8. BERT-SSIN (Fine-Tuning)

### Cài đặt
```bash
pip install transformers scikit-learn spacy
python -m spacy download en_core_web_sm
```

### Đánh giá
Mở và chạy toàn bộ notebook trên Kaggle (yêu cầu GPU T4):

```
notebooks/Ssin_sameval.ipynb
```

Kết quả được lưu tại:
```
/kaggle/working/bert_ssin_restaurant_metrics.json
/kaggle/working/bert_ssin_laptop_metrics.json
```

## 9. Tổng hợp kết quả

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
