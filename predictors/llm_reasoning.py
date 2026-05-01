from __future__ import annotations

import sys
import json
from pathlib import Path

# Thêm đường dẫn để import code thực nghiệm thực tế từ models/syn-chain-LLM
SYN_CHAIN_DIR = Path(__file__).parent.parent / "models" / "syn-chain-LLM"
sys.path.append(str(SYN_CHAIN_DIR))

from syn_chain_pipeline import run_syn_chain
from syntax_parser import get_syntactic_dependency_string

PARSE_ERROR = "__PARSE_ERROR__"

class LLMReasoningPredictor:
    method   = "Syn-Chain"
    paradigm = "LLM-Reasoning"
    backbone = "Qwen-2.5-14B-Instruct"

    def __init__(self, **kwargs):
        # Thông tin cấu hình (API KEY, BASE URL, MODEL) 
        # đã được tự động load từ file .env bên trong syn_chain_pipeline.py
        pass

    def warmup(self, text: str) -> None:
        pass

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        if not aspect:
            # Syn-Chain hiện tại yêu cầu phải có aspect (given-aspect)
            return PARSE_ERROR, PARSE_ERROR, "Syn-Chain requires an explicit aspect."
            
        try:
            # Bước 1: Trích xuất cú pháp
            dep_seq = get_syntactic_dependency_string(text)
            
            # Bước 2: Chạy 3 chain LLM reasoning (Cú pháp -> Quan điểm -> Cảm xúc)
            result = run_syn_chain(text, aspect, dep_seq)
            
            # Phân tích kết quả bước cuối
            pred_raw = result['step3_sentiment'].lower()
            if "positive" in pred_raw:
                pred_sentiment = "positive"
            elif "negative" in pred_raw:
                pred_sentiment = "negative"
            elif "neutral" in pred_raw:
                pred_sentiment = "neutral"
            else:
                pred_sentiment = PARSE_ERROR
                
            raw_output = json.dumps(result, ensure_ascii=False)
            return aspect, pred_sentiment, raw_output
            
        except Exception as exc:  # noqa: BLE001
            return PARSE_ERROR, PARSE_ERROR, f"<<error: {exc}>>"
