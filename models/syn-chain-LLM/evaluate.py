import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from syntax_parser import get_syntactic_dependency_string
from syn_chain_pipeline import run_syn_chain

def evaluate(data_path: str, limit: int = None, output_log_path: str = "evaluation_logs.json"):
    # Đọc dữ liệu test (định dạng JSON)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if limit:
        data = data[:limit]
        print(f"Giới hạn {limit} ví dụ")
        
    y_true = []
    y_pred = []
    logs = []
    
    # Duyệt qua từng câu
    for item in tqdm(data, desc="Evaluating Syn-Chain"):
        text = item['text']
        aspect = item['aspect']
        true_sentiment = item['sentiment'].lower()
        
        # 1. Phân tích cú pháp với SpaCy
        dep_seq = get_syntactic_dependency_string(text)
        
        # 2. Chạy chuỗi suy luận 3 bước với LLM
        try:
            result = run_syn_chain(text, aspect, dep_seq)
            pred_raw = result['step3_sentiment'].lower()
            
            # Chuẩn hóa kết quả dự đoán của LLM
            if "positive" in pred_raw:
                pred = "positive"
            elif "negative" in pred_raw:
                pred = "negative"
            else:
                pred = "neutral"
                
            y_true.append(true_sentiment)
            y_pred.append(pred)
            
            # Lưu lại log suy luận
            logs.append({
                "text": text,
                "aspect": aspect,
                "ground_truth": true_sentiment,
                "prediction": pred,
                "llm_reasoning": result
            })
        except Exception as e:
            print(f"\nLỗi khi xử lý câu '{text}': {e}")
            continue

    # Đánh giá Metrics
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        print("\n" + "="*30)
        print("KẾT QUẢ ĐÁNH GIÁ TẬP TEST")
        print("="*30)
        print(f"Tổng số mẫu đánh giá : {len(y_true)}")
        print(f"Accuracy             : {acc:.4f}")
        print(f"Macro-F1             : {f1:.4f}")
        print("="*30)
        
        with open(output_log_path, "w", encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu logs suy luận chi tiết của LLM tại: {output_log_path}")
    else:
        print("Không có mẫu nào được đánh giá thành công.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá phương pháp Syn-Chain trên SemEval-2014")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn tới file JSON dữ liệu test")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng mẫu để chạy thử")
    parser.add_argument("--out", type=str, default="evaluation_logs.json", help="File lưu output logs")
    args = parser.parse_args()
    
    evaluate(args.data, args.limit, args.out)
