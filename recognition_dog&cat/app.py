from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os


app = Flask(__name__)

# Sử dụng đường dẫn tương đối để dễ bảo trì và tương thích với các hệ thống khác
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dogcats.h5")

# Tải model
model = load_model(MODEL_PATH, compile=False)

# Thư mục lưu ảnh tải lên
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_image(file_path):
    """Dự đoán loại ảnh (Dog, Cat, Horse) dựa trên model."""
    try:
        img = load_img(file_path, target_size=(150, 150))
        img = img_to_array(img) / 255.0  # Chuẩn hóa ảnh
        img = np.expand_dims(img, axis=0)  # Thêm batch dimension

        # Dự đoán lớp ảnh
        prediction = model.predict(img)
        classes = ['Cat', 'Dog', 'Horse']
        predicted_class = classes[np.argmax(prediction)]  # Trả về lớp có xác suất cao nhất
        return predicted_class
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    """Trang chủ với chức năng tải ảnh và dự đoán."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400  # Không có tệp nào được tải lên

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400  # Không có tệp nào được chọn

        # Lưu tệp vào thư mục chỉ định
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        print(f"Tệp đã lưu tại: {file_path}")  # Kiểm tra xem tệp đã được lưu

        # Gọi hàm dự đoán
        result = predict_image(file_path)
        if result:
            return jsonify({
                'prediction': result,
                'file_path': url_for('download_file', filename=file.filename)  # Đường dẫn tải về
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500  # Dự đoán thất bại

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Phục vụ tệp đã tải lên từ thư mục UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
