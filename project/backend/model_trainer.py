# Bước 4: Huấn luyện mô hình (Model Training)
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle

class ModelTrainer:
    """
    Lớp thực hiện huấn luyện mô hình machine learning
    Bước 4 trong quy trình xử lý ngôn ngữ tự nhiên
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_times = {}
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành tập train và test
        """
        try:
            print(f"Chia dữ liệu: {X.shape[0]} mẫu")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y  # Đảm bảo tỷ lệ nhãn đều
            )
            
            print(f"Train set: {X_train.shape[0]} mẫu")
            print(f"Test set: {X_test.shape[0]} mẫu")
            print(f"Train spam ratio: {y_train.sum() / len(y_train):.3f}")
            print(f"Test spam ratio: {y_test.sum() / len(y_test):.3f}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Lỗi chia dữ liệu: {e}")
            raise e
    
    def select_model(self, method_name, X_shape):
        """
        Chọn mô hình phù hợp với loại dữ liệu
        """
        if method_name == 'Sentence Embeddings':
            # GaussianNB cho continuous features (embeddings)
            model = GaussianNB()
        else:
            # MultinomialNB cho discrete features (BoW, TF-IDF)
            model = MultinomialNB(alpha=1.0)  # Laplace smoothing
        
        print(f"Chọn mô hình: {type(model).__name__} cho {method_name}")
        return model
    
    def train_single_model(self, X, y, method_name, socket_callback=None):
        """
        Huấn luyện một mô hình đơn lẻ
        """
        start_time = time.time()
        try:
            if socket_callback:
                socket_callback('training_progress', {
                    'method': method_name,
                    'status': 'splitting_data',
                    'message': f'Chia dữ liệu cho {method_name}...'
                })
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            if socket_callback:
                socket_callback('training_progress', {
                    'method': method_name,
                    'status': 'training',
                    'message': f'Huấn luyện mô hình {method_name}...'
                })
            # Chọn và huấn luyện mô hình
            model = self.select_model(method_name, X.shape)
            model.fit(X_train, y_train)
            
            if socket_callback:
                socket_callback('training_progress', {
                    'method': method_name,
                    'status': 'evaluating',
                    'message': f'Đánh giá mô hình {method_name}...'
                })
            # Dự đoán và đánh giá
            y_pred = model.predict(X_test)
            # Tính các metrics
            accuracy = accuracy_score(y_test, y_pred)
            # output_dict=True để lấy chi tiết macro avg
            report = classification_report(y_test, y_pred, output_dict=True) 
            cm = confusion_matrix(y_test, y_pred)
            training_time = time.time() - start_time
            # Cross-validation để đánh giá độ ổn định
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            result = {
                'model': model,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_pred': y_pred,
                'training_time': training_time,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            # Lưu trữ kết quả
            self.models[method_name.lower().replace(' ', '_')] = model
            self.results[method_name.lower().replace(' ', '_')] = result
            self.training_times[method_name.lower().replace(' ', '_')] = training_time
            
            # SỬA ĐỔI: Sử dụng 'macro avg' thay cho 'weighted avg' trong log
            print(f"✅ {method_name} - Accuracy: {accuracy:.4f}, Macro F1: {report['macro avg']['f1-score']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Lỗi huấn luyện {method_name}: {e}")
            raise e
    
    def train_all_models(self, features_dict, labels, socket_callback=None):
        """
        Huấn luyện tất cả mô hình với các loại features khác nhau
        """
        all_results = {}
        
        for method_name, features in features_dict.items():
            try:
                print(f"\n🚀 Bắt đầu huấn luyện {method_name}...")
                result = self.train_single_model(features, labels, method_name, socket_callback)
                
                # Chuẩn bị kết quả cho frontend
                method_key = method_name.lower().replace(' ', '_')
                all_results[method_key] = {
                    'name': method_name,
                    'accuracy': float(result['accuracy']),
                    # SỬA ĐỔI: Sử dụng 'macro avg'
                    'precision': float(result['classification_report']['macro avg']['precision']),
                    'recall': float(result['classification_report']['macro avg']['recall']),
                    'f1_score': float(result['classification_report']['macro avg']['f1-score']),
                    'training_time': result['training_time'],
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
                
                if socket_callback:
                    socket_callback('model_completed', {
                        'method': method_key,
                        'results': all_results[method_key]
                    })
                    
            except Exception as e:
                print(f"❌ {method_name} thất bại: {e}")
                if socket_callback:
                    socket_callback('model_error', {
                        'method': method_name,
                        'error': str(e)
                    })
                continue
        
        return all_results
    
    def get_best_model(self, metric='f1_score'):
        """
        Tìm mô hình tốt nhất dựa trên metric
        """
        if not self.results:
            return None
        
        best_method = None
        best_score = -1
        
        for method, result in self.results.items():
            # SỬA ĐỔI: Sử dụng 'macro avg' cho các metrics
            if metric == 'f1_score':
                score = result['classification_report']['macro avg']['f1-score']
            elif metric == 'accuracy':
                score = result['accuracy']
            elif metric == 'precision':
                score = result['classification_report']['macro avg']['precision']
            elif metric == 'recall':
                score = result['classification_report']['macro avg']['recall']
            else:
                score = result['accuracy']
            
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method, best_score
    
    def predict_new_data(self, X_new, method='best'):
        """
        Dự đoán dữ liệu mới
        """
        if method == 'best':
            method, _ = self.get_best_model()
        
        if method not in self.models:
            raise ValueError(f"Mô hình {method} không tồn tại")
        
        model = self.models[method]
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def save_models(self, filepath):
        """
        Lưu tất cả mô hình vào file
        """
        try:
            model_package = {
                'models': self.models,
                'results': self.results,
                'training_times': self.training_times
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f"Đã lưu mô hình vào {filepath}")
            
        except Exception as e:
            print(f"Lỗi lưu mô hình: {e}")
    
    def load_models(self, filepath):
        """
        Tải mô hình từ file
        """
        try:
            with open(filepath, 'rb') as f:
                model_package = pickle.load(f)
            
            self.models = model_package['models']
            self.results = model_package['results']
            self.training_times = model_package['training_times']
            
            print(f"Đã tải mô hình từ {filepath}")
            
        except Exception as e:
            print(f"Lỗi tải mô hình: {e}")
    
    def get_training_summary(self):
        """
        Tóm tắt quá trình huấn luyện
        """
        if not self.results:
            return None
        
        summary = {
            'total_models': len(self.results),
            'total_training_time': sum(self.training_times.values()),
            'models': {}
        }
        
        for method, result in self.results.items():
            summary['models'][method] = {
                'accuracy': result['accuracy'],
                # SỬA ĐỔI: Sử dụng 'macro avg'
                'f1_score': result['classification_report']['macro avg']['f1-score'],
                'training_time': self.training_times[method],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
        
        return summary