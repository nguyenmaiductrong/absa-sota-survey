# InstructABSA vs LCF-BERT — kết quả thực nghiệm

| Dataset | Method | Backbone | sent_acc | macro_F1 | latency (ms) | parse_err |
|---|---|---|---:|---:|---:|---:|
| SemEval-14 Restaurant (en) | LCF-BERT | bert-base-uncased | 0.8941 | 0.8232 | 20.80 | 0.000 |
| SemEval-14 Restaurant (en) | InstructABSA | checkpoints/semeval14_rest/instruct_absa_best | 0.8767 | 0.7718 | 61.91 | 0.000 |
| SemEval-14 Laptop (en) | LCF-BERT | bert-base-uncased | 0.7962 | 0.7516 | 21.35 | 0.000 |
| SemEval-14 Laptop (en) | InstructABSA | checkpoints/semeval14_lap/instruct_absa_best | 0.7837 | 0.7278 | 63.72 | 0.000 |
| UIT-VSFC (vi) | LCF-BERT | vinai/phobert-base | 0.9274 | 0.8121 | 20.30 | 0.000 |
| UIT-VSFC (vi) | InstructABSA | checkpoints/vsfc/instruct_absa_best | 0.6453 | 0.4252 | 74.14 | 0.000 |
