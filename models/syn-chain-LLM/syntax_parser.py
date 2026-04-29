import spacy
import pandas as pd

# Load or download the spaCy english model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model 'en_core_web_sm' for the spaCy POS tagger...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_syntactic_dependency_string(text: str) -> str:
    """
    Phân tích cú pháp văn bản và trả về string dạng bảng mô tả các quan hệ phụ thuộc cú pháp.
    """
    doc = nlp(text)
    data = []
    for token in doc:
        data.append({
            "ID": token.i + 1,
            "Word": token.text,
            "Head": token.head.text,
            "DepRel": token.dep_
        })
    df = pd.DataFrame(data)
    return df.to_string(index=False)

if __name__ == "__main__":
    # Test block
    sample = "The food was great but the service was terrible."
    print(f"Testing on: {sample}\n")
    print(get_syntactic_dependency_string(sample))
