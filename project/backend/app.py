# Main Application - Khởi tạo Flask app và Socket.IO
from flask import Flask
from flask_socketio import SocketIO
from api_handler import APIHandler

# Khởi tạo Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'spam_classifier_secret_key'

# Khởi tạo SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')  

# Khởi tạo API Handler
api_handler = APIHandler(app, socketio)

if __name__ == '__main__':
    print("🚀 Khởi động hệ thống phân loại Spam SMS...")
    print("📊 5 bước quy trình xử lý ngôn ngữ tự nhiên:")
    print("   1. Khám phá dữ liệu (data_explorer.py)")
    print("   2. Tiền xử lý văn bản (text_preprocessor.py)")
    print("   3. Vector hóa văn bản (text_vectorizer.py)")
    print("   4. Huấn luyện mô hình (model_trainer.py)")
    print("   5. Đánh giá kết quả (result_analyzer.py)")
    print("🌐 API Handler: api_handler.py")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)