# API Handler - Xử lý các API endpoints và Socket.IO events
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import io
import base64
import json
import time
import glob, os
from datetime import datetime

# Import các module xử lý
from data_explorer import DataExplorer
from text_preprocessor import TextPreprocessor
from text_vectorizer import TextVectorizer
from model_trainer import ModelTrainer
from result_analyzer import ResultAnalyzer


class APIHandler:
    """
    Lớp xử lý các API endpoints và Socket.IO events
    Điều phối các bước xử lý ngôn ngữ tự nhiên
    """

    def __init__(self, app, socketio):
        self.app = app
        self.socketio = socketio

        # Khởi tạo các module xử lý
        self.data_explorer = DataExplorer()
        self.text_preprocessor = TextPreprocessor()
        self.text_vectorizer = TextVectorizer()
        self.model_trainer = ModelTrainer()
        self.result_analyzer = ResultAnalyzer()

        # Lưu trữ dữ liệu
        self.original_data = None
        self.processed_data = None
        self.start_time = None

        # Đăng ký các routes và events
        self.register_routes()
        self.register_socket_events()

    def register_routes(self):
        """Đăng ký các HTTP routes"""

        @self.app.route('/')
        def index():
            return render_template('index.html')

    def register_socket_events(self):
        """Đăng ký các Socket.IO events"""

        @self.socketio.on('connect')
        def handle_connect():
            print('Client đã kết nối')
            emit('connected', {'message': 'Kết nối thành công!'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client đã ngắt kết nối')

        @self.socketio.on('upload_file')
        def handle_file_upload(data):
            self.handle_file_upload(data)

        @self.socketio.on('process_data')
        def handle_process_data():
            self.handle_process_data()

        @self.socketio.on('get_confusion_matrix')
        def handle_get_confusion_matrix(data):
            self.handle_get_confusion_matrix(data)

    def handle_file_upload(self, data):
        """
        Xử lý việc tải file CSV từ client
        Bước 1: Khám phá dữ liệu
        """
        try:
            filename = data['filename']
            file_content = data['file_content']

            if not filename.lower().endswith('.csv'):
                emit('upload_error', {'message': 'Chỉ chấp nhận file CSV'})
                return

            # Giải mã file
            file_data = base64.b64decode(file_content.split(',')[1])

            # Thử các encoding khác nhau
            content_str = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content_str = file_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if content_str is None:
                emit('upload_error', {'message': 'Không thể đọc file'})
                return

            # Parse CSV
            df = None
            for sep in [',', '\t', ';', '|']:
                try:
                    df_temp = pd.read_csv(io.StringIO(content_str), sep=sep)
                    if len(df_temp.columns) >= 2:
                        df = df_temp
                        break
                except:
                    continue

            if df is None or df.empty:
                emit('upload_error', {'message': 'Không thể phân tích file CSV'})
                return

            # Tự động nhận diện cột
            df = self.auto_detect_columns(df)

            # Chuẩn hóa nhãn
            df = self.normalize_labels(df)

            if df.empty:
                emit('upload_error', {'message': 'Không có dữ liệu hợp lệ'})
                return

            # Lưu dữ liệu gốc
            self.original_data = df

            # Bước 1: Khám phá dữ liệu
            self.data_explorer.load_data(df)
            stats = self.data_explorer.get_basic_statistics()

            # Tạo biểu đồ phân phối
            distribution_chart = self.data_explorer.generate_label_distribution_chart()

            # Phân tích độ dài văn bản
            text_length_analysis = self.data_explorer.analyze_text_length()

            # Gửi kết quả thành công
            emit('upload_success', {
                'message': 'Tải file thành công!',
                'filename': filename,
                'stats': stats,
                'sample_data': df.head().fillna('').to_dict('records'),
                'distribution_chart': distribution_chart,
                'text_length_analysis': text_length_analysis
            })

        except Exception as e:
            print(f"Lỗi upload: {str(e)}")
            emit('upload_error', {'message': f'Lỗi tải file: {str(e)}'})

    def auto_detect_columns(self, df):
        """Tự động nhận diện cột label và message"""
        print(f"Cột gốc: {list(df.columns)}")

        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].copy()
            df.columns = ['label', 'message']
        elif 'label' in df.columns and 'message' in df.columns:
            df = df[['label', 'message']].copy()
        elif 'sms' in df.columns and 'label' in df.columns:
            df = df[['label', 'sms']].copy()
            df.columns = ['label', 'message']
        else:
            # Tự động phát hiện
            if len(df.columns) < 2:
                raise ValueError('File CSV cần ít nhất 2 cột')

            # Sử dụng 2 cột đầu tiên
            df = df.iloc[:, :2].copy()
            df.columns = ['label', 'message']

        return df.dropna()

    def normalize_labels(self, df):
        """Chuẩn hóa nhãn về 0 (ham) và 1 (spam)"""
        original_labels = df['label'].unique()
        print(f"Nhãn gốc: {original_labels}")

        label_mapping = {}

        for label in original_labels:
            label_str = str(label).lower().strip()

            if label_str in ['ham', '0'] or label == 0:
                label_mapping[label] = 0
            elif label_str in ['spam', '1'] or label == 1:
                label_mapping[label] = 1
            else:
                if 'ham' in label_str or 'normal' in label_str:
                    label_mapping[label] = 0
                elif 'spam' in label_str:
                    label_mapping[label] = 1
                else:
                    raise ValueError(f'Nhãn không xác định: "{label}"')

        df['label'] = df['label'].map(label_mapping)
        return df.dropna()

    def handle_process_data(self):
        """
        Xử lý dữ liệu và huấn luyện mô hình
        Thực hiện các bước 2, 3, 4, 5
        """
        try:
            if self.original_data is None:
                emit('process_error', {'message': 'Chưa có dữ liệu'})
                return

            self.start_time = time.time()
            df = self.original_data.copy()

            # Bước 2: Tiền xử lý văn bản
            emit('step_update', {'step': 2, 'status': 'active', 'message': 'Bắt đầu tiền xử lý...'})

            def progress_callback(event, data):
                self.socketio.emit(event, data)

            processed_texts = self.text_preprocessor.preprocess_batch(
                df['message'].tolist(),
                socket_callback=progress_callback
            )

            df['processed_message'] = processed_texts
            df = df[df['processed_message'].str.len() > 0]

            if df.empty:
                emit('process_error', {'message': 'Không có văn bản hợp lệ sau tiền xử lý'})
                return

            self.processed_data = df
            emit('step_update', {'step': 2, 'status': 'completed', 'message': 'Tiền xử lý hoàn thành!'})

            # Bước 3: Vector hóa
            emit('step_update', {'step': 3, 'status': 'active', 'message': 'Bắt đầu vector hóa...'})

            processed_texts = df['processed_message'].tolist()
            features_dict = {}

            # Tạo BoW features
            try:
                bow_features, bow_vectorizer = self.text_vectorizer.create_bow_features(processed_texts)
                features_dict['Bag of Words'] = bow_features
            except Exception as e:
                print(f"Lỗi BoW: {e}")

            # Tạo TF-IDF features
            try:
                tfidf_features, tfidf_vectorizer = self.text_vectorizer.create_tfidf_features(processed_texts)
                features_dict['TF-IDF'] = tfidf_features
            except Exception as e:
                print(f"Lỗi TF-IDF: {e}")

            # Tạo Sentence Embeddings (tùy chọn)
            try:
                sentence_embeddings, sentence_model = self.text_vectorizer.create_sentence_embeddings(processed_texts)
                features_dict['Sentence Embeddings'] = sentence_embeddings
            except Exception as e:
                print(f"Lỗi Sentence Embeddings: {e}")

            emit('step_update', {'step': 3, 'status': 'completed', 'message': 'Vector hóa hoàn thành!'})

            # Bước 4: Huấn luyện mô hình
            emit('step_update', {'step': 4, 'status': 'active', 'message': 'Bắt đầu huấn luyện...'})

            results = self.model_trainer.train_all_models(
                features_dict,
                df['label'],
                socket_callback=progress_callback
            )

            emit('step_update', {'step': 4, 'status': 'completed', 'message': 'Huấn luyện hoàn thành!'})

            # Bước 5: Phân tích kết quả
            self.result_analyzer.set_results(
                self.model_trainer.results,
                self.text_vectorizer.vectorizers
            )

            total_time = time.time() - self.start_time

            # Tạo các biểu đồ phân tích
            best_method, _ = self.model_trainer.get_best_model()

            confusion_matrix_img = self.result_analyzer.generate_confusion_matrix(best_method)
            comparison_chart = self.result_analyzer.generate_comparison_chart()
            feature_analysis = self.result_analyzer.generate_feature_analysis()
            summary = self.result_analyzer.generate_summary(total_time)

            # Tạo file competition (sử dụng file test có sẵn)
            competition_data = self.generate_competition_files(best_method)
            if competition_data:
                self.socketio.emit('test_files_ready', competition_data)

            # Gửi kết quả hoàn chỉnh
            emit('process_completed', {
                'message': 'Xử lý hoàn thành!',
                'results': results,
                'best_method': best_method,
                'confusion_matrix': confusion_matrix_img,
                'comparison_chart': comparison_chart,
                'feature_analysis': feature_analysis,
                'summary': summary
            })

        except Exception as e:
            print(f"Lỗi process: {e}")
            emit('process_error', {'message': f'Lỗi xử lý: {str(e)}'})

    def generate_competition_files(self, best_method):
        """
        Sử dụng file test có sẵn (test.csv hoặc file tên chứa 'test') để dự đoán,
        sinh ra submission.csv cho frontend tải.
        """
        try:
            # 1) Tìm file test
            candidates = []
            candidates += glob.glob('test*.csv') + glob.glob('*test*.csv')
            candidates += glob.glob('/mnt/data/test*.csv') + glob.glob('/mnt/data/*test*.csv')
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                print("Không tìm thấy file test (*.csv)")
                return None

            chosen = None
            for p in candidates:
                if os.path.basename(p).lower() == 'test.csv':
                    chosen = p
                    break
            if chosen is None:
                chosen = candidates[0]
            print(f"Found test file: {chosen}")

            # 2) Đọc file test
            test_df = pd.read_csv(chosen)
            if 'sms' not in test_df.columns:
                if 'message' in test_df.columns:
                    test_df = test_df.rename(columns={'message': 'sms'})
                elif test_df.shape[1] >= 2:
                    col0, col1 = test_df.columns[:2]
                    test_df = test_df.rename(columns={col0: 'id', col1: 'sms'})
            if 'id' not in test_df.columns:
                test_df = test_df.reset_index().rename(columns={'index': 'id'})

            # 3) Tiền xử lý
            raw_sms = test_df['sms'].astype(str).tolist()
            processed = self.text_preprocessor.preprocess_batch(raw_sms)
            final_texts = [p if p and len(p.strip()) > 0 else s for p, s in zip(processed, raw_sms)]

            # 4) Chọn vectorizer key
            bm = str(best_method).lower()
            if bm in self.text_vectorizer.vectorizers:
                vec_key = bm
            elif 'bag' in bm or 'bow' in bm:
                vec_key = 'bow'
            elif 'tf' in bm:
                vec_key = 'tfidf'
            elif 'sentence' in bm:
                vec_key = 'sentence'
            else:
                vec_key = next(iter(self.text_vectorizer.vectorizers.keys()))

            # 5) Dự đoán
            features = self.text_vectorizer.transform_new_texts(final_texts, vec_key)
            predictions, _ = self.model_trainer.predict_new_data(features, method=best_method)
            preds = list(map(int, predictions))

            submission_df = pd.DataFrame({'id': test_df['id'].values, 'label': preds})

            return {
                'test_csv': test_df.to_csv(index=False),
                'submission_csv': submission_df.to_csv(index=False),
                'test_filename': os.path.basename(chosen),
                'submission_filename': f'submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                'count': len(preds)
            }

        except Exception as e:
            print(f"Lỗi tạo file competition: {e}")
            return None
