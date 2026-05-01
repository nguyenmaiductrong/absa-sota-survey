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

## 4. Hướng dẫn chạy thử nghiệm
### 4.1. LCF-BERT (Discriminative - 2019)
### 4.2. InstructABSA (Instruction-Tuning - 2023)
### 4.3. SSIN (Graph-based - 2024)
### 4.4. DOT (Generative-Seq2Seq - 2025)

### 4.5. LLM-Reasoning (LLM-Based - 2025)

