# Bước 1: Khám phá dữ liệu (Data Exploration)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from collections import Counter

class DataExplorer:
    """
    Lớp thực hiện khám phá và phân tích dữ liệu
    Bước 1 trong quy trình xử lý ngôn ngữ tự nhiên
    """
    
    def __init__(self):
        self.data = None
        self.stats = {}
    
    def load_data(self, df):
        """Tải dữ liệu và lưu trữ"""
        self.data = df.copy()
        return self.data
    
    def get_basic_statistics(self):
        """
        Tính toán các thống kê cơ bản về dữ liệu
        - Số lượng mẫu tổng cộng
        - Phân phối các nhãn (ham/spam)
        - Tỷ lệ phần trăm từng loại
        """
        if self.data is None:
            return None
            
        total_messages = len(self.data)
        spam_count = int(self.data['label'].sum())
        ham_count = int(total_messages - spam_count)
        spam_percentage = (spam_count / total_messages * 100) if total_messages > 0 else 0.0
        
        self.stats = {
            'total_messages': total_messages,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'spam_percentage': round(spam_percentage, 2)
        }
        
        return self.stats
    
    def analyze_text_length(self):
        """
        Phân tích độ dài văn bản
        So sánh độ dài trung bình giữa tin nhắn ham và spam
        """
        if self.data is None:
            return None
            
        # Tính độ dài cho từng tin nhắn
        self.data['text_length'] = self.data['message'].astype(str).str.len()
        
        # Phân tích riêng cho ham và spam
        ham_data = self.data[self.data['label'] == 0]['text_length']
        spam_data = self.data[self.data['label'] == 1]['text_length']
        
        ham_stats = {
            'mean': float(ham_data.mean()),
            'median': float(ham_data.median()),
            'min': int(ham_data.min()),
            'max': int(ham_data.max()),
            'std': float(ham_data.std())
        }
        
        spam_stats = {
            'mean': float(spam_data.mean()),
            'median': float(spam_data.median()),
            'min': int(spam_data.min()),
            'max': int(spam_data.max()),
            'std': float(spam_data.std())
        }
        
        # Tạo biểu đồ so sánh
        chart_image = self.create_length_comparison_chart(ham_data, spam_data)
        
        return {
            'ham_stats': ham_stats,
            'spam_stats': spam_stats,
            'chart': chart_image
        }
    
    def create_length_comparison_chart(self, ham_data, spam_data):
        """Tạo biểu đồ histogram so sánh độ dài văn bản"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Tạo subplot cho histogram
            plt.subplot(1, 2, 1)
            plt.hist([ham_data, spam_data], bins=30, alpha=0.7, 
                    label=['Ham', 'Spam'], color=['#48bb78', '#e53e3e'])
            plt.xlabel('Độ dài văn bản (ký tự)')
            plt.ylabel('Tần suất')
            plt.title('Phân phối Độ dài Văn bản')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Tạo subplot cho box plot
            plt.subplot(1, 2, 2)
            data_to_plot = [ham_data, spam_data]
            box_plot = plt.boxplot(data_to_plot, labels=['Ham', 'Spam'], 
                                  patch_artist=True)
            box_plot['boxes'][0].set_facecolor('#48bb78')
            box_plot['boxes'][1].set_facecolor('#e53e3e')
            plt.ylabel('Độ dài văn bản (ký tự)')
            plt.title('Box Plot Độ dài Văn bản')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Chuyển thành base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Lỗi tạo biểu đồ độ dài: {e}")
            return None
    
    def generate_label_distribution_chart(self):
        """
        Tạo biểu đồ tròn phân phối nhãn
        Xác định vấn đề mất cân bằng lớp
        """
        try:
            if not self.stats:
                self.get_basic_statistics()
                
            labels = ['Ham (Hợp lệ)', 'Spam (Rác)']
            counts = [self.stats['ham_count'], self.stats['spam_count']]
            colors = ['#48bb78', '#e53e3e']
            
            plt.figure(figsize=(10, 8))
            
            # Biểu đồ tròn
            wedges, texts, autotexts = plt.pie(counts, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 12, 'fontweight': 'bold'})
            plt.title('Phân phối Nhãn - Ham vs Spam', fontsize=16, fontweight='bold', pad=20)
            
            # Làm đẹp text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
            
            # Thêm thông tin phân tích mất cân bằng
            imbalance_ratio = max(counts) / min(counts)
            imbalance_text = f"Tỷ lệ mất cân bằng: {imbalance_ratio:.1f}:1"
            
            if imbalance_ratio > 3:
                imbalance_status = "⚠️ Dữ liệu mất cân bằng nghiêm trọng"
                color_status = 'red'
            elif imbalance_ratio > 2:
                imbalance_status = "⚠️ Dữ liệu mất cân bằng vừa phải"
                color_status = 'orange'
            else:
                imbalance_status = "✅ Dữ liệu tương đối cân bằng"
                color_status = 'green'
            
            # Text box thông tin
            textstr = f'{imbalance_text}\n{imbalance_status}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.figtext(0.5, 0.02, textstr, fontsize=12, ha='center', va='bottom',
                       bbox=props, color=color_status, fontweight='bold')
            
            plt.tight_layout()
            
            # Chuyển thành base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150,
                       facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Lỗi tạo biểu đồ phân phối: {e}")
            return None
    
    def detect_data_issues(self):
        """
        Phát hiện các vấn đề tiềm ẩn trong dữ liệu
        - Dữ liệu thiếu (missing values)
        - Dữ liệu trùng lặp
        - Văn bản quá ngắn hoặc quá dài
        """
        if self.data is None:
            return None
            
        issues = {
            'missing_values': int(self.data.isnull().sum().sum()),
            'duplicate_messages': int(self.data.duplicated(subset=['message']).sum()),
            'empty_messages': int((self.data['message'].astype(str).str.len() == 0).sum()),
            'very_short_messages': int((self.data['message'].astype(str).str.len() < 5).sum()),
            'very_long_messages': int((self.data['message'].astype(str).str.len() > 500).sum())
        }
        
        return issues
    
    def get_sample_data(self, n=5):
        """Lấy dữ liệu mẫu để hiển thị"""
        if self.data is None:
            return None
            
        return self.data.head(n).fillna('').to_dict('records')