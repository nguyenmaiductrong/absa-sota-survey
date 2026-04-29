import os
import json
import csv
import xml.etree.ElementTree as ET

# Lấy thư mục gốc của project (nằm trên 2 level so với file hiện tại)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "syn-chain")

def process_semeval_xml(filepath, output_filepath):
    if not os.path.exists(filepath):
        print(f"File không tồn tại: {filepath}")
        return
        
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    all_data = []
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                term = aspect_term.get('term')
                polarity = aspect_term.get('polarity')
                
                if polarity in ['positive', 'negative', 'neutral']:
                    all_data.append({
                        "text": text,
                        "aspect": term,
                        "sentiment": polarity
                    })
                    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(all_data)} examples to {output_filepath}")

def process_uit_vsfc_csv(filepath, output_filepath):
    if not os.path.exists(filepath):
        print(f"File không tồn tại: {filepath}")
        return
        
    topic_mapping = {
        '0': 'giảng viên',
        '1': 'chương trình giảng dạy',
        '2': 'cơ sở vật chất',
        '3': 'khác'
    }
    
    sentiment_mapping = {
        '0': 'negative',
        '1': 'neutral',
        '2': 'positive'
    }
    
    all_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row.get('sentence', '').strip()
            sentiment_idx = row.get('sentiment', '').strip()
            topic_idx = row.get('topic', '').strip()
            
            if not sentence or not sentiment_idx or not topic_idx:
                continue
                
            sentiment = sentiment_mapping.get(sentiment_idx)
            aspect = topic_mapping.get(topic_idx)
            
            if sentiment and aspect:
                all_data.append({
                    "text": sentence,
                    "aspect": aspect,
                    "sentiment": sentiment
                })
                
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(all_data)} examples to {output_filepath}")

def process_and_save():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Processing raw datasets...")
    
    # Laptops
    laptops_raw = os.path.join(RAW_DATA_DIR, "semeval14", "Laptops_Test_Gold.xml")
    laptops_out = os.path.join(OUTPUT_DIR, "laptops_test.json")
    print(f"Processing {laptops_raw}...")
    process_semeval_xml(laptops_raw, laptops_out)
    
    # Restaurants
    rest_raw = os.path.join(RAW_DATA_DIR, "semeval14", "Restaurants_Test_Gold.xml")
    rest_out = os.path.join(OUTPUT_DIR, "restaurants_test.json")
    print(f"Processing {rest_raw}...")
    process_semeval_xml(rest_raw, rest_out)
    
    # UIT-VSFC
    vsfc_raw = os.path.join(RAW_DATA_DIR, "uit-vsfc", "test.csv")
    vsfc_out = os.path.join(OUTPUT_DIR, "uit-vsfc_test.json")
    print(f"Processing {vsfc_raw}...")
    process_uit_vsfc_csv(vsfc_raw, vsfc_out)

def main():
    process_and_save()

if __name__ == "__main__":
    main()
