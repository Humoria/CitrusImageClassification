from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model_jeruk.h5')  # pastikan file ini ada di folder yang sama

# class names sesuai label model training
class_names = ['bali', 'lemon', 'limau', 'mandarin']  # SESUAIKAN dengan model kamu

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    # Simpan gambar
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Proses gambar
    img = image.load_img(filepath, target_size=(224, 224))  # sesuaikan ukuran sesuai training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    if class_index < len(class_names):
        class_label = class_names[class_index]
    else:
        class_label = "Kelas tidak dikenal"

    return render_template('result.html', label=class_label, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
