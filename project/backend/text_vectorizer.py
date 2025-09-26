# Bước 3: Vector hóa văn bản (Text Vectorization)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

class TextVectorizer:
    """
    Lớp thực hiện vector hóa văn bản
    Bước 3 trong quy trình xử lý ngôn ngữ tự nhiên
    """
    
    def __init__(self):
        self.vectorizers = {}
        self.sentence_model = None
    
    def create_bow_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Tạo đặc trưng Bag of Words (BoW)
        - Đếm tần suất xuất hiện của từ
        - Sử dụng n-gram để capture cụm từ
        """
        try:
            print(f"Tạo BoW features cho {len(texts)} văn bản...")
            if not texts or len(texts) == 0:
                raise ValueError("Danh sách văn bản rỗng")
            
            # Lọc bỏ các văn bản rỗng hoặc None
            valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                raise ValueError("Không có văn bản hợp lệ sau khi lọc")
            
            print(f"Số văn bản hợp lệ: {len(valid_texts)}")
            print(f"Mẫu văn bản: {valid_texts[0][:100] if valid_texts else 'None'}")
            
            vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            
            features = vectorizer.fit_transform(valid_texts)
            if features.shape[0] == 0:
                raise ValueError("BoW features rỗng sau khi vector hóa")
            
            print(f"BoW features shape: {features.shape}")
            print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
            
            # Lưu vectorizer
            self.vectorizers['bow'] = vectorizer
            print("✅ BoW vectorizer đã được lưu")
            
            return features, vectorizer
            
        except Exception as e:
            print(f"❌ Lỗi tạo BoW features: {e}")
            raise e
    
    def create_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Tạo đặc trưng TF-IDF
        - TF: Term Frequency
        - IDF: Inverse Document Frequency
        """
        try:
            print(f"Tạo TF-IDF features cho {len(texts)} văn bản...")
            if not texts or len(texts) == 0:
                raise ValueError("Danh sách văn bản rỗng")
            
            # Lọc bỏ các văn bản rỗng hoặc None
            valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                raise ValueError("Không có văn bản hợp lệ sau khi lọc")
            
            print(f"Số văn bản hợp lệ: {len(valid_texts)}")
            print(f"Mẫu văn bản: {valid_texts[0][:100] if valid_texts else 'None'}")
            
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                stop_words='english',
                sublinear_tf=True
            )
            
            features = vectorizer.fit_transform(valid_texts)
            if features.shape[0] == 0:
                raise ValueError("TF-IDF features rỗng sau khi vector hóa")
            
            print(f"TF-IDF features shape: {features.shape}")
            print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
            
            # Lưu vectorizer
            self.vectorizers['tfidf'] = vectorizer
            print("✅ TF-IDF vectorizer đã được lưu")
            
            return features, vectorizer
            
        except Exception as e:
            print(f"❌ Lỗi tạo TF-IDF features: {e}")
            raise e
    
    def load_sentence_model(self):
        """
        Tải mô hình Sentence Transformer
        """
        try:
            if self.sentence_model is None:
                print("Đang tải mô hình sentence transformer...")
                
                models_to_try = [
                    'all-MiniLM-L6-v2',
                    'paraphrase-MiniLM-L6-v2',
                    'all-mpnet-base-v2'
                ]
                
                for model_name in models_to_try:
                    try:
                        print(f"Đang thử tải: {model_name}")
                        from sentence_transformers import SentenceTransformer
                        self.sentence_model = SentenceTransformer(model_name)
                        print(f"✅ Tải thành công: {model_name}")
                        return self.sentence_model
                    except Exception as e:
                        print(f"Lỗi tải {model_name}: {e}")
                        continue
                
                raise ValueError("Không thể tải bất kỳ mô hình sentence transformer nào")
                
        except Exception as e:
            print(f"❌ Lỗi tải sentence model: {e}")
            raise e
    
    def create_sentence_embeddings(self, texts, batch_size=32):
        """
        Tạo sentence embeddings sử dụng SentenceTransformer
        """
        try:
            print(f"Tạo sentence embeddings cho {len(texts)} văn bản...")
            if not texts or len(texts) == 0:
                raise ValueError("Danh sách văn bản rỗng")
            
            # Lọc bỏ các văn bản rỗng hoặc None
            valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                raise ValueError("Không có văn bản hợp lệ sau khi lọc")
            
            print(f"Số văn bản hợp lệ: {len(valid_texts)}")
            print(f"Mẫu văn bản: {valid_texts[0][:100] if valid_texts else 'None'}")
            
            model = self.load_sentence_model()
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(valid_texts) + batch_size - 1) // batch_size
                
                print(f"Xử lý batch {batch_num}/{total_batches}")
                
                try:
                    batch_embeddings = model.encode(
                        batch, 
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"❌ Lỗi batch {batch_num}: {e}")
                    dummy_embeddings = np.zeros((len(batch), 384))
                    all_embeddings.extend(dummy_embeddings)
            
            embeddings = np.array(all_embeddings)
            if embeddings.shape[0] == 0:
                raise ValueError("Sentence embeddings rỗng sau khi encode")
            
            print(f"Embeddings shape: {embeddings.shape}")
            
            # Lưu model
            self.vectorizers['sentence'] = model
            print("✅ Sentence model đã được lưu")
            
            return embeddings, model
            
        except Exception as e:
            print(f"❌ Lỗi tạo sentence embeddings: {e}")
            raise e
    
    def transform_new_texts(self, texts, method='bow'):
        """
        Transform văn bản mới sử dụng vectorizer đã fit
        """
        try:
            print(f"Transform văn bản mới với phương pháp: {method}")
            print(f"Vectorizers hiện có: {list(self.vectorizers.keys())}")
            if method not in self.vectorizers:
                raise ValueError(f"Vectorizer {method} chưa được tạo")
            
            # Lọc bỏ các văn bản rỗng hoặc None
            valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                raise ValueError("Không có văn bản hợp lệ để transform")
            
            print(f"Số văn bản hợp lệ để transform: {len(valid_texts)}")
            print(f"Mẫu văn bản: {valid_texts[0][:100] if valid_texts else 'None'}")
            
            vectorizer = self.vectorizers[method]
            
            if method in ['bow', 'tfidf']:
                features = vectorizer.transform(valid_texts)
                print(f"Features shape sau transform ({method}): {features.shape}")
                return features
            elif method == 'sentence':
                features = vectorizer.encode(valid_texts, convert_to_numpy=True)
                print(f"Embeddings shape sau transform: {features.shape}")
                return features
            else:
                raise ValueError(f"Phương pháp {method} không được hỗ trợ")
        
        except Exception as e:
            print(f"❌ Lỗi transform văn bản mới: {e}")
            raise e
    
    def get_feature_names(self, method='bow'):
        """
        Lấy tên các features (chỉ cho BoW và TF-IDF)
        """
        if method not in ['bow', 'tfidf']:
            return None
        
        if method not in self.vectorizers:
            return None
        
        vectorizer = self.vectorizers[method]
        return vectorizer.get_feature_names_out()
    
    def get_vocabulary_stats(self, method='bow'):
        """
        Thống kê về vocabulary
        """
        if method not in ['bow', 'tfidf']:
            return None
        
        if method not in self.vectorizers:
            return None
        
        vectorizer = self.vectorizers[method]
        vocab = vectorizer.vocabulary_
        
        stats = {
            'vocabulary_size': len(vocab),
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
            'min_df': vectorizer.min_df,
            'max_df': vectorizer.max_df
        }
        
        return stats
    
    def save_vectorizers(self, filepath):
        """
        Lưu tất cả vectorizers vào file
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.vectorizers, f)
            print(f"Đã lưu vectorizers vào {filepath}")
        except Exception as e:
            print(f"Lỗi lưu vectorizers: {e}")
    
    def load_vectorizers(self, filepath):
        """
        Tải vectorizers từ file
        """
        try:
            with open(filepath, 'rb') as f:
                self.vectorizers = pickle.load(f)
            print(f"Đã tải vectorizers từ {filepath}")
        except Exception as e:
            print(f"Lỗi tải vectorizers: {e}")