# Hệ thống Phân loại Spam SMS

Hệ thống phân loại tin nhắn SMS spam/ham sử dụng Machine Learning với giao diện web real-time.

## 🚀 Tính năng chính

- **Khám phá dữ liệu**: Phân tích thống kê, biểu đồ phân phối nhãn
- **Tiền xử lý văn bản**: Làm sạch, tách từ, loại bỏ stopwords
- **Vector hóa**: Bag-of-Words, TF-IDF, Sentence Embeddings
- **Machine Learning**: Naive Bayes với nhiều phương pháp
- **Đánh giá mô hình**: Accuracy, Precision, Recall, F1-Score
- **Trực quan hóa**: Biểu đồ so sánh, ma trận nhầm lẫn
- **Real-time**: Giao diện web với Socket.IO
- **Competition**: Tự động tạo file test.csv và submission.csv

## 📋 Yêu cầu hệ thống

- Python 3.8+
- 4GB RAM (khuyến nghị 8GB cho Sentence Embeddings)
- 2GB dung lượng ổ cứng

## ⚡ Cài đặt nhanh

### Phương pháp 1: Tự động (Khuyến nghị)
```bash
python install.py
```

### Phương pháp 2: Thủ công
```bash
pip install -r requirements.txt
```

## 🎯 Chạy hệ thống

1. **Khởi động server:**
```bash
cd backend
python app.py
```

2. **Mở trình duyệt:**
```
http://localhost:8000
```

3. **Sử dụng:**
   - Tải file CSV (định dạng: label, message)
   - Nhấn "Bắt đầu Xử lý"
   - Xem kết quả real-time

## 📊 Định dạng dữ liệu

### File CSV đầu vào:
```csv
label,message
ham,"How are you today?"
spam,"WINNER! You have won $1000! Call now!"
```

### Các định dạng được hỗ trợ:
- `v1,v2` (SMS Spam Collection)
- `label,message` 
- `label,sms`
- Tự động phát hiện cột

## 🔧 Cấu trúc thư viện

### Thư viện cần thiết:
- **Flask + SocketIO**: Web framework và real-time
- **pandas + numpy**: Xử lý dữ liệu
- **scikit-learn**: Machine learning
- **nltk**: Xử lý ngôn ngữ tự nhiên
- **matplotlib + seaborn**: Trực quan hóa

### Thư viện tùy chọn:
- **sentence-transformers**: Sentence embeddings
- **torch**: Deep learning backend

## 📈 Quy trình xử lý

1. **Khám phá dữ liệu**
   - Thống kê cơ bản
   - Phân phối nhãn (biểu đồ tròn)
   - Phân tích độ dài văn bản

2. **Tiền xử lý**
   - Lowercase, loại bỏ ký tự đặc biệt
   - Tách từ, loại bỏ stopwords
   - Chuẩn hóa khoảng trắng

3. **Vector hóa**
   - Bag-of-Words (n-gram 1,2)
   - TF-IDF (n-gram 1,2)
   - Sentence Embeddings (SBERT)

4. **Huấn luyện**
   - MultinomialNB (BoW, TF-IDF)
   - GaussianNB (Sentence Embeddings)
   - Train/Test split 80/20

5. **Đánh giá**
   - Accuracy, Precision, Recall, F1
   - Confusion Matrix
   - Feature Importance
   - So sánh các phương pháp

## 🏆 File Competition

Hệ thống tự động tạo:
- **test.csv**: Dữ liệu test (id, sms)
- **submission.csv**: Kết quả dự đoán (id, label)

## 🐛 Xử lý lỗi

### Lỗi thường gặp:

1. **ModuleNotFoundError**:
```bash
pip install -r requirements.txt
```

2. **NLTK Data Error**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

3. **Memory Error** (Sentence Embeddings):
   - Giảm batch_size trong code
   - Hoặc bỏ qua Sentence Embeddings

4. **Port 8000 đã được sử dụng**:
   - Thay đổi port trong `app.py`: `port=8001`

---
*Được phát triển bằng Python + Flask + Machine Learning*
