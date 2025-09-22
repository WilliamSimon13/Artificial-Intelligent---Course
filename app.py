from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load model MNIST
model = tf.keras.models.load_model("mnist_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if request.method == 'POST':
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        # Lưu ảnh tạm
        with open('temp.jpg', 'wb') as f:
            f.write(imgBytes)

        # Đọc ảnh, xử lý
        img = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = 255 - img        # đảo ngược (trắng/đen)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Dự đoán
        prediction = model.predict(img)
        digit = int(np.argmax(prediction))

        return jsonify({
            'prediction': str(digit),
            'status': True
        })

if __name__ == '__main__':
    app.run(debug=True)
