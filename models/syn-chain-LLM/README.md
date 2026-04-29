# Syn-Chain ABSA Pipeline

Dự án này triển khai phương pháp Syn-Chain (Syntactic Dependency Chain) để giải quyết bài toán Aspect-Based Sentiment Analysis (ABSA). Pipeline bao gồm 3 bước suy luận liên tiếp (Phân tích cú pháp, Trích xuất quan điểm, và Dự đoán cảm xúc) sử dụng các mô hình ngôn ngữ lớn (LLMs).

## 🚀 1. Cài đặt Môi trường

Dự án yêu cầu **Python 3.9+**. Để tránh xung đột thư viện, bạn nên tạo một môi trường ảo (virtual environment).

### Khởi tạo và kích hoạt môi trường ảo (venv)

**Trên Windows:**
```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\activate
```

**Trên macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Cài đặt thư viện phụ thuộc

Sau khi đã kích hoạt môi trường ảo, hãy cài đặt các thư viện được liệt kê trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Lưu ý: Mô hình SpaCy `en_core_web_sm` dùng để phân tích cú pháp sẽ tự động được tải xuống trong lần đầu tiên bạn chạy code nếu chưa có sẵn).*

---

## ⚙️ 2. Cấu hình Biến Môi Trường (.env)

Trong thư mục gốc của dự án, bạn sẽ thấy file `.env`. File này lưu trữ cấu hình liên kết đến LLM server (ví dụ: Qwen, Llama chạy qua vLLM/Ollama, hoặc OpenAI) và cấu hình LangSmith để theo dõi các node (trace).

Cấu trúc file `.env` cơ bản:

```env
# Cấu hình API LLM (Mặc định dùng model local hoặc qua Ngrok proxy)
QWEN_API_BASE=https://<your-ngrok-url>.ngrok-free.dev
MODEL_NAME=qwen2.5-7b-instruct
# QWEN_API_KEY= # Bỏ comment và điền nếu server của bạn yêu cầu API key

# --- Cấu hình LangSmith ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Syn_Chain_ABSA
```

**Lưu ý:**
1. Hãy cập nhật `QWEN_API_BASE` thành địa chỉ API server LLM của bạn.
2. Nếu bạn muốn theo dõi quá trình chạy và đánh giá output từng node, hãy thay `your_langsmith_api_key_here` bằng [LangSmith API Key](https://smith.langchain.com/) thực tế của bạn.

---

## 💾 3. Chuẩn bị Dữ liệu

Dự án sử dụng bộ dữ liệu **SemEval-2014** (từ thư viện `datasets` của HuggingFace). Để tải và tiền xử lý dữ liệu chuẩn bị cho quá trình đánh giá, bạn chạy script sau:

```bash
python data_processor.py
```

**Kết quả:** Script sẽ tự động kết nối với HuggingFace, tải dataset `NEUDM/semeval-2014`, trích xuất các câu có chứa khía cạnh (aspect) cùng với nhãn cảm xúc, và lưu dưới dạng 2 file JSON:
- `data/train.json`
- `data/test.json`

---

## 🧠 4. Chạy Đánh Giá (Evaluation)

Sau khi đã chuẩn bị xong dữ liệu, bạn có thể chạy file `evaluate.py` để LLM thực hiện dự đoán dựa trên phương pháp Syn-Chain và đo lường độ chính xác (Accuracy, Macro-F1).

```bash
# Chạy đánh giá toàn bộ tập test
python evaluate.py --data data/test.json

# (Tuỳ chọn) Chạy đánh giá trên 10 mẫu đầu tiên để kiểm tra
python evaluate.py --data data/test.json --limit 10
```

**Các tham số bổ sung:**
- `--out`: Tên file (JSON) lưu chi tiết quá trình suy luận và đối chiếu kết quả (Mặc định: `evaluation_logs.json`).
  Ví dụ: `python evaluate.py --data data/test.json --limit 10 --out my_results.json`

---

## 📊 5. Theo dõi (Tracing) từng Node trên LangSmith

Nếu bạn đã thiết lập `LANGCHAIN_API_KEY` trong file `.env`, toàn bộ quá trình xử lý mỗi câu sẽ tự động được gửi lên LangSmith dưới dạng các luồng suy luận.

1. Đăng nhập vào [LangSmith](https://smith.langchain.com/).
2. Chọn Project có tên **Syn_Chain_ABSA** (như cấu hình trong file `.env`).
3. Trong các lần chạy (Runs), bạn sẽ thấy các Trace lớn có tên **Syn-Chain Pipeline**.
4. Bấm vào mỗi Trace, bạn sẽ thấy chính xác 3 bước (Nodes) đang diễn ra:
   - Node 1: *Phân tích cú pháp* (LLM nhận Dependency Sequence và diễn dịch)
   - Node 2: *Trích xuất quan điểm* (LLM lọc từ ngữ mô tả)
   - Node 3: *Dự đoán cảm xúc* (LLM chốt phân loại Positive/Negative/Neutral)

Bạn có thể dễ dàng phân tích input/output tại mỗi node để tinh chỉnh (fine-tune) prompt nếu cần!
