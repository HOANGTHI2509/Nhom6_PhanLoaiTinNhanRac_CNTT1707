# Bước 5: Đánh giá và phân tích kết quả (Result Analysis)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

class ResultAnalyzer:
    """
    Lớp thực hiện đánh giá và phân tích kết quả
    Bước 5 trong quy trình xử lý ngôn ngữ tự nhiên
    """
    
    def __init__(self):
        self.results = {}
        self.vectorizers = {}
        
    def set_results(self, results, vectorizers=None):
        """Thiết lập kết quả để phân tích"""
        self.results = results
        self.vectorizers = vectorizers or {}
    
    def generate_confusion_matrix(self, method):
        """
        Tạo ma trận nhầm lẫn cho một mô hình
        """
        try:
            if method not in self.results:
                return None
            
            result = self.results[method]
            cm = result['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'])
            plt.title(f'Ma trận Nhầm lẫn - {method.upper()}', fontsize=14, fontweight='bold')
            plt.ylabel('Nhãn Thực tế')
            plt.xlabel('Nhãn Dự đoán')
            
            # Thêm thông tin chi tiết
            tn, fp, fn, tp = cm.ravel()
            plt.figtext(0.02, 0.02, f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}', 
                       fontsize=10, ha='left')
            
            plt.tight_layout()
            
            # Chuyển thành base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Lỗi tạo confusion matrix: {e}")
            return None
    
    def generate_comparison_chart(self):
        """
        Tạo biểu đồ so sánh hiệu suất các mô hình
        """
        try:
            if not self.results:
                return None
            
            methods = list(self.results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            axes = [ax1, ax2, ax3, ax4]
            
            colors = ['#667eea', '#764ba2', '#48bb78', '#f093fb']
            
            for i, metric in enumerate(metrics):
                # Lấy giá trị cho từng method
                values = []
                method_names = []
                
                for method in methods:
                    if method in self.results:
                        result = self.results[method]
                        if metric == 'accuracy':
                            values.append(result['accuracy'] * 100)
                        else:
                            values.append(result['classification_report']['weighted avg'][metric.replace('_', '-')] * 100)
                        method_names.append(result.get('name', method).replace('_', ' ').title())
                
                if values:
                    bars = axes[i].bar(method_names, values, color=colors[:len(values)])
                    axes[i].set_title(f'{metric.replace("_", " ").title()} (%)', 
                                     fontsize=14, fontweight='bold')
                    axes[i].set_ylim(0, 100)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Thêm nhãn giá trị
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Chuyển thành base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Lỗi tạo biểu đồ so sánh: {e}")
            return None
    
    def generate_feature_analysis(self):
        """
        Phân tích tầm quan trọng của features (cho BoW)
        """
        try:
            if 'bow' not in self.results or 'bow' not in self.vectorizers:
                return None
            
            model = self.results['bow']['model']
            vectorizer = self.vectorizers['bow']
            
            # Lấy feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Lấy log probabilities
            spam_features = model.feature_log_prob_[1]  # Spam class
            ham_features = model.feature_log_prob_[0]   # Ham class
            
            # Tính feature importance
            feature_importance = spam_features - ham_features
            
            # Top 15 features quan trọng nhất
            top_indices = np.argsort(feature_importance)[-15:]
            top_features = [feature_names[i] for i in top_indices]
            top_scores = [feature_importance[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(top_features))
            
            bars = plt.barh(y_pos, top_scores, color='#e53e3e', alpha=0.7)
            plt.yticks(y_pos, top_features)
            plt.xlabel('Mức độ quan trọng cho SPAM detection')
            plt.title('Top 15 từ quan trọng cho SPAM detection - BoW', 
                     fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Thêm giá trị
            for i, (bar, score) in enumerate(zip(bars, top_scores)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            
            # Chuyển thành base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Lỗi tạo feature analysis: {e}")
            return None
    
    def generate_performance_report(self):
        """
        Tạo báo cáo hiệu suất chi tiết
        """
        try:
            if not self.results:
                return None
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'summary': {}
            }
            
            # Phân tích từng mô hình
            for method, result in self.results.items():
                model_report = {
                    'accuracy': float(result['accuracy']),
                    'precision': float(result['classification_report']['weighted avg']['precision']),
                    'recall': float(result['classification_report']['weighted avg']['recall']),
                    'f1_score': float(result['classification_report']['weighted avg']['f1-score']),
                    'training_time': float(result['training_time']),
                    'cross_validation': {
                        'mean': float(result['cv_mean']),
                        'std': float(result['cv_std'])
                    },
                    'confusion_matrix': result['confusion_matrix'].tolist(),
                    'classification_report': result['classification_report']
                }
                
                report['models'][method] = model_report
            
            # Tóm tắt
            best_f1_method = max(self.results.keys(), 
                               key=lambda x: self.results[x]['classification_report']['weighted avg']['f1-score'])
            best_accuracy_method = max(self.results.keys(), 
                                     key=lambda x: self.results[x]['accuracy'])
            
            report['summary'] = {
                'total_models': len(self.results),
                'best_f1_model': best_f1_method,
                'best_f1_score': float(self.results[best_f1_method]['classification_report']['weighted avg']['f1-score']),
                'best_accuracy_model': best_accuracy_method,
                'best_accuracy_score': float(self.results[best_accuracy_method]['accuracy']),
                'total_training_time': sum(result['training_time'] for result in self.results.values())
            }
            
            return report
            
        except Exception as e:
            print(f"Lỗi tạo báo cáo: {e}")
            return None
    
    def generate_summary(self, total_time):
        """
        Tạo tóm tắt kết quả cho giao diện
        """
        try:
            if not self.results:
                return None
            
            # Tìm mô hình tốt nhất
            best_method = max(self.results.keys(), 
                            key=lambda x: self.results[x]['classification_report']['weighted avg']['f1-score'])
            best_result = self.results[best_method]
            
            f1_score = best_result['classification_report']['weighted avg']['f1-score']
            
            # Đánh giá hiệu suất
            if f1_score >= 0.95:
                performance = "Xuất sắc! 🏆"
            elif f1_score >= 0.90:
                performance = "Rất tốt! 🎯"
            elif f1_score >= 0.85:
                performance = "Tốt! 👍"
            else:
                performance = "Cần cải thiện 📈"
            
            summary = {
                'best_model': best_method.replace('_', ' ').title(),
                'best_f1': f1_score,
                'total_models': len(self.results),
                'total_time': total_time,
                'performance': performance,
                'training_times': {method: result['training_time'] 
                                 for method, result in self.results.items()}
            }
            
            return summary
            
        except Exception as e:
            print(f"Lỗi tạo tóm tắt: {e}")
            return None
    
    def export_results(self, filepath):
        """
        Xuất kết quả ra file JSON
        """
        try:
            import json
            
            report = self.generate_performance_report()
            if report:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Đã xuất kết quả ra {filepath}")
            
        except Exception as e:
            print(f"Lỗi xuất kết quả: {e}")