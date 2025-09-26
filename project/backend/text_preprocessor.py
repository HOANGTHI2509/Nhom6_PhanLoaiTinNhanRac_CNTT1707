# Bước 2: Tiền xử lý văn bản (Text Preprocessing)
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """
    Lớp xử lý tiền xử lý văn bản cho hệ thống phân loại spam SMS
    Bước 2 trong quy trình xử lý ngôn ngữ tự nhiên
    """
    
    def __init__(self):
        # Khởi tạo và tải dữ liệu NLTK cần thiết
        self.setup_nltk()
        self.english_stopwords = set(stopwords.words('english'))
    
    def setup_nltk(self):
        """Tải dữ liệu NLTK cần thiết"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Đang tải punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Đang tải stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text):
        """
        Làm sạch và chuẩn hóa văn bản đầu vào
        - Chuyển về chữ thường
        - Loại bỏ ký tự đặc biệt, số, dấu câu
        - Chuẩn hóa khoảng trắng
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Chuyển thành string và lowercase
        text = str(text).lower()
        
        # Loại bỏ các khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Bỏ qua văn bản quá ngắn
        if len(text) < 2:
            return ""
        
        # Chỉ giữ lại chữ cái và khoảng trắng
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Loại bỏ từ dừng (stop words) và từ quá ngắn
        """
        if not text or len(text.strip()) < 2:
            return ""
        
        try:
            # Sử dụng NLTK tokenizer
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"Lỗi NLTK tokenize: {e}")
            # Fallback: split đơn giản
            tokens = text.split()
        
        # Lọc bỏ stopwords và từ quá ngắn
        filtered_tokens = [
            token for token in tokens 
            if token not in self.english_stopwords and len(token) > 1
        ]
        
        return ' '.join(filtered_tokens)
    
    def normalize_text(self, text):
        """
        Chuẩn hóa văn bản nâng cao
        - Xử lý từ viết tắt
        - Chuẩn hóa số điện thoại, email
        - Xử lý từ lặp lại
        """
        if not text:
            return ""
        
        # Chuẩn hóa từ viết tắt phổ biến
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Thay thế số điện thoại bằng token
        text = re.sub(r'\b\d{3,}\b', 'NUMBER', text)
        
        # Thay thế email bằng token
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        
        # Thay thế URL bằng token
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        
        # Loại bỏ từ lặp lại (ví dụ: "hahaha" -> "haha")
        text = re.sub(r'\b(\w+)\1+\b', r'\1', text)
        
        return text
    
    def preprocess_single_text(self, text):
        """
        Tiền xử lý một văn bản đơn lệ
        """
        # Bước 1: Làm sạch cơ bản
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        
        # Bước 2: Chuẩn hóa nâng cao
        normalized = self.normalize_text(cleaned)
        if not normalized:
            return cleaned
        
        # Bước 3: Loại bỏ stopwords
        final_text = self.remove_stopwords(normalized)
        
        return final_text if final_text else cleaned
    
    def preprocess_batch(self, texts, socket_callback=None):
        """
        Tiền xử lý một batch văn bản với callback tiến trình
        """
        processed_texts = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                processed_text = self.preprocess_single_text(text)
                processed_texts.append(processed_text)
                
                # Gửi cập nhật tiến trình mỗi 50 văn bản
                if socket_callback and (i + 1) % 50 == 0:
                    progress = (i + 1) / total * 100
                    socket_callback('preprocessing_progress', {
                        'progress': progress,
                        'current': i + 1,
                        'total': total,
                        'message': f'Đã xử lý {i + 1}/{total} tin nhắn'
                    })
                    
            except Exception as e:
                print(f"Lỗi xử lý văn bản {i}: {e}")
                # Giữ nguyên văn bản gốc nếu có lỗi
                processed_texts.append(str(text) if text else "")
        
        return processed_texts
    
    def get_preprocessing_stats(self, original_texts, processed_texts):
        """
        Thống kê quá trình tiền xử lý
        """
        original_lengths = [len(str(text)) for text in original_texts]
        processed_lengths = [len(str(text)) for text in processed_texts]
        
        stats = {
            'original_avg_length': np.mean(original_lengths),
            'processed_avg_length': np.mean(processed_lengths),
            'reduction_ratio': 1 - (np.mean(processed_lengths) / np.mean(original_lengths)),
            'empty_after_processing': sum(1 for text in processed_texts if len(str(text).strip()) == 0),
            'total_processed': len(processed_texts)
        }
        
        return stats
    
    def validate_preprocessing(self, processed_texts):
        """
        Kiểm tra chất lượng tiền xử lý
        """
        issues = {
            'empty_texts': sum(1 for text in processed_texts if not str(text).strip()),
            'very_short_texts': sum(1 for text in processed_texts if len(str(text).strip()) < 3),
            'potential_over_processing': sum(1 for text in processed_texts if len(str(text).split()) < 2)
        }
        
        return issues