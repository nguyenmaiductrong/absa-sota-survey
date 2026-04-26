# ABSA-SOTA-Survey: Khảo sát Trade-off của 5 Paradigm trên SemEval-2014 & UIT-VSFC

Repository thực nghiệm 5 phương pháp ABSA SOTA (2019-2025) trên cả tiếng Anh
(SemEval-2014) và tiếng Việt (UIT-VSFC), tập trung phân tích trade-off
**Accuracy vs Time** và khả năng generalize xuyên ngôn ngữ.

## 📋 Mục lục
- [1. Phương pháp khảo sát](#1)
- [2. Dataset](#2)
- [3. Cấu trúc thư mục](#3)

---

## 1. Phương pháp khảo sát 

| # | Method | Year | Paradigm | Backbone EN | Backbone VI |
|---|--------|------|----------|-------------|-------------|
| 1 | LCF-BERT | 2019 | Discriminative | bert-base-uncased | vinai/phobert-base |
| 2 | InstructABSA | 2023 | Instruction Tuning | allenai/tk-instruct-base-def-pos | google/mt5-base |
| 3 | SSIN | 2024 | Graph (Syn+Sem) | bert-base + spaCy | phobert-base + VnCoreNLP |
| 4 | DOT | 2025 | Generative Seq2Seq | t5-base | VietAI/vit5-base |
| 5 | LLM-Reasoning | 2025 | LLM + QLoRA + CoT | meta-llama/Llama-3-8B | SeaLLMs/SeaLLM-7B-v2 |

## 2. Dataset 

### SemEval-2014 (English)
- **Task**: ATSC — Aspect Term Sentiment Classification
- **Domain**: Restaurant + Laptop reviews
- **Size**: ~6,000 câu (sau khi loại Conflict)
- **Nhãn sentiment**: Positive / Negative / Neutral

### UIT-VSFC (Vietnamese) — "đơn nghĩa"
- **Task**: ACSA — Aspect Category Sentiment Analysis (single-label per sentence)
- **Domain**: Vietnamese student feedback (giáo dục)
- **Size**: ~16,175 câu
- **Nhãn topic**: lecturer (0) / training_program (1) / facility (2) / others (3)
- **Nhãn sentiment**: negative (0) / neutral (1) / positive (2)
- **Đặc điểm**: mỗi câu chỉ có duy nhất 1 (topic, sentiment) → ACSA single-label

---

## 3. Cấu trúc thư mục 

```
absa-sota-survey/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── semeval14/                  # 4 file XML
│   │   └── vsfc/                       # train/dev/test (sents.txt, sentiments.txt, topics.txt)
│   ├── processed/
│   │   ├── en/                         # rest14, lap14 đã chuẩn hoá JSON
│   │   └── vi/                         # vsfc đã word-segment + JSON
│   └── formatted/                      # định dạng riêng từng model × ngôn ngữ
│       ├── lcf_bert/{en,vi}/
│       ├── instruct_absa/{en,vi}/
│       ├── ssin/{en,vi}/
│       ├── dot/{en,vi}/
│       └── llm_reasoning/{en,vi}/
│
├── models/
│   ├── lcf_bert/      {model.py, train.py, infer.py}
│   ├── instruct_absa/
│   ├── ssin/
│   ├── dot/
│   └── llm_reasoning/
│
├── scripts/
│   ├── preprocess_semeval.py
│   ├── preprocess_vsfc.py              # MỚI - word segment + format
│   ├── format_data.py --model X --lang {en,vi}
│   ├── benchmark_latency.py
│   ├── aggregate_results.py
│   └── plot_tradeoff.py
│
├── configs/
│   ├── lcf_bert_en.yaml
│   ├── lcf_bert_vi.yaml                # separate config per language
│   ├── ...
│   └── llm_reasoning_vi.yaml
│
├── results/
│   ├── checkpoints/
│   ├── metrics/
│   │   ├── summary_en.csv
│   │   ├── summary_vi.csv
│   │   └── summary_all.csv             # merged
│   └── figures/
│       ├── pareto_en_latency_f1.pdf
│       ├── pareto_vi_latency_f1.pdf
│       ├── pareto_combined.pdf       
│       ├── train_time_comparison.pdf
│       └── f1_drop_heatmap.pdf
│
└── notebooks/
```
