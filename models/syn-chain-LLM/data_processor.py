import os
import json
import ast
from datasets import load_dataset

DATA_DIR = "data"

def process_and_save():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print("Đang tải dataset NEUDM/semeval-2014...")
    ds = load_dataset("NEUDM/semeval-2014")
    
    for split in ['train', 'test']:
        data = []
        for row in ds[split]:
            text = row['input'][0] if isinstance(row['input'], list) else row['input']
            try:
                output_obj = ast.literal_eval(row['output'])
            except Exception:
                continue
                
            if isinstance(output_obj, dict):
                aspects = output_obj.get('aspect_term', [])
            elif isinstance(output_obj, list):
                aspects = output_obj
            else:
                aspects = []
                
            for aspect_item in aspects:
                if isinstance(aspect_item, list) and len(aspect_item) == 2:
                    aspect, polarity = aspect_item
                    if polarity in ['positive', 'negative', 'neutral']:
                        data.append({
                            "text": text,
                            "aspect": aspect,
                            "sentiment": polarity.lower()
                        })
                        
        json_filepath = os.path.join(DATA_DIR, f"{split}.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(data)} examples to {json_filepath}")

def main():
    process_and_save()

if __name__ == "__main__":
    main()
