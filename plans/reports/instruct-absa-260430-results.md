# InstructABSA vs LCF-BERT — kết quả thực nghiệm

| Dataset | Method | Backbone | sent_acc | macro_F1 | latency (ms) | parse_err |
|---|---|---|---:|---:|---:|---:|
| SemEval-14 Restaurant (en) | LCF-BERT | bert-base-uncased | 0.8941 | 0.8232 | 20.80 | 0.000 |
| SemEval-14 Restaurant (en) | InstructABSA | allenai/tk-instruct-base-def-pos | 0.8931 | 0.8086 | 53.47 | 0.000 |
| SemEval-14 Laptop (en) | LCF-BERT | bert-base-uncased | 0.7962 | 0.7516 | 21.35 | 0.000 |
| SemEval-14 Laptop (en) | InstructABSA | allenai/tk-instruct-base-def-pos | 0.7994 | 0.7520 | 53.94 | 0.000 |
| UIT-VSFC (vi) | LCF-BERT | vinai/phobert-base | 0.9274 | 0.8121 | 20.30 | 0.000 |
| UIT-VSFC (vi) | InstructABSA | VietAI/vit5-base | 0.9321 | 0.8109 | 65.52 | 0.000 |

**Cấu hình InstructABSA:** 100% training data, fp16, batch 4, grad accum 4, 4 epochs, T4 16GB. Backbone VI: VietAI/vit5-base (T5 pretrain trên 138GB text Việt thuần).
