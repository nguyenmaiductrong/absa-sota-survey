# ABSA-SOTA-Survey: Khảo sát Trade-off của 5 Paradigm trên SemEval-2014 & UIT-VSFC

Repository thực nghiệm 5 phương pháp ABSA SOTA (2019–2025) trên tiếng Anh (SemEval-2014) và tiếng Việt (UIT-VSFC), tập trung phân tích trade-off **Accuracy vs Latency** và khả năng generalize đa ngôn ngữ.

---

## 1. Phương pháp khảo sát

| # | Method | Year | Paradigm | Backbone EN | Backbone VI |
|---|--------|------|----------|-------------|-------------|
| 1 | LCF-BERT | 2019 | Discriminative | `bert-base-uncased` | PhoBERT (`vinai/phobert-base`) |
| 2 | InstructABSA | 2024 | Instruction-Tuning | `allenai/tk-instruct-base-def-pos` | `VietAI/vit5-base` |
| 3 | SSIN | 2024 | Graph (Syn+Sem) | `bert-base-uncased` + spaCy | PhoBERT-base + `underthesea` (tách từ) |
| 4 | DOT | 2025 | Generative-Seq2Seq | `t5-base` | `VietAI/vit5-base` |
| 5 | Syn-Chain (LLM) | 2025 | LLM-Reasoning | Qwen 2.5 14B Instruct | Qwen 2.5 14B Instruct |

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

Các chỉ số dưới đây thống nhất với pipeline `evaluate.py` + module chấm [`eval/score.py`](eval/score.py): mỗi dòng trong file dự đoán (JSONL) là một **mẫu ATSC** — cùng một câu đánh giá với **aspect hoặc topic đã cho** (`given_aspect`), mô hình chỉ dự đoán **cực cảm xúc** (subtask *Aspect (-Category) Sentiment Classification*).

| Metric | Ý nghĩa (định nghĩa dùng trong repo) |
|--------|--------------------------------------|
| `sentiment_accuracy` | **Accuracy** (độ chính xác thường): tỷ số mẫu mà nhãn dự đoán `pred.sentiment` **trùng khớp** nhãn tham chiếu `gold.sentiment` trên toàn bộ tập đánh giá (\(\#\) đúng / \(\#\) mẫu). Giá trị thực trong \([0, 1]\); khi báo cáo %, nhân \(100\). *Không* phải F1 hay precision/recall riêng lẻ. |
| `sentiment_macro_f1` | **Macro-F1** (*macro-averaged F1*): với **mỗi lớp cực cảm xúc** xuất hiện trong tập `gold`, tính F1 (harmonic mean của precision và recall lớp đó), rồi lấy **trung bình cộng** các F1 lớp. Nhạy với lớp thiểu số hơn accuracy; phù hợp SemEval khi `neutral` ít hơn `positive`. Triển khai: trung bình không trọng số theo lớp trong `eval/score.py`. |
| `avg_latency_ms` | **Độ trễ suy luận trung bình** (*end-to-end latency*): trung bình số học của `latency_ms` trên **tất cả** mẫu test; mỗi `latency_ms` là **thời gian tường (wall-clock)** của **một lần gọi** dự đoán (forward / giải mã, có thể nhiều vòng với LLM). Các lần **warmup** trước vòng đo không tính vào danh sách latency. Đơn vị: millisecond. |

---

## 4. Hướng dẫn chạy trên Kaggle (notebook)

Thử nghiệm nằm trong [`notebook/`](notebook/). Trên Kaggle: tạo notebook → **File** → **Import Notebook** (dán URL file `.ipynb` trên GitHub nhánh `main`, dạng `https://github.com/nguyenmaiductrong/absa-sota-survey/blob/main/notebook/<tên-file>.ipynb`) hoặc làm theo cell clone/thiết lập đầu tiên của từng file. Bật **Internet** / **GPU** khi notebook có ghi chú. **Chạy tuần tự các cell**; cài đặt, dataset, secrets và đường dẫn `/kaggle/...` đều được mô tả **trong notebook**.

### 4.1. LCF-BERT

Chạy theo [`notebook/01-lcf-bert-train-eval.ipynb`](notebook/01-lcf-bert-train-eval.ipynb).

### 4.2. InstructABSA

Chạy theo [`notebook/02-instruct-absa-train-eval.ipynb`](notebook/02-instruct-absa-train-eval.ipynb).

### 4.3. SSIN

Chạy theo [`notebook/03-Notebook_SSIN_train_SAMEVAL.ipynb`](notebook/03-Notebook_SSIN_train_SAMEVAL.ipynb) (SemEval-2014) và/hoặc [`notebook/03-Notebook_SSIN_train_UIT.ipynb`](notebook/03-Notebook_SSIN_train_UIT.ipynb) (UIT-VSFC), tùy bộ dữ liệu cần tái hiện.

### 4.4. DOT

Chạy theo [`notebook/04-dot-train-restaurant.ipynb`](notebook/04-dot-train-restaurant.ipynb), [`notebook/04-dot-train-laptop.ipynb`](notebook/04-dot-train-laptop.ipynb) và [`notebook/04-eval-dot-sem.ipynb`](notebook/04-eval-dot-sem.ipynb) — thứ tự, checkpoint và đánh giá nằm trong từng notebook.

### 4.5. Syn-Chain (LLM)

Chạy theo [`notebook/05-nlp-survey-syn-chain-absa.ipynb`](notebook/05-nlp-survey-syn-chain-absa.ipynb) (SemEval, EN) và/hoặc [`notebook/05-nlp-survey-syn-chain-absa-vi.ipynb`](notebook/05-nlp-survey-syn-chain-absa-vi.ipynb) (UIT-VSFC, VI).

---
