# Orange Image Classification using CNN 🍊

This project is a deep learning-based image classification system designed to identify **four types of oranges** using Convolutional Neural Networks (CNNs). It was developed as part of a computer vision coursework assignment using TensorFlow and Keras.

## 🧠 Project Goal

To build a CNN-based classifier that can accurately recognize and differentiate between the following types of oranges:
- Jeruk Mandarin
- Jeruk Bali
- Jeruk Lemon
- Jeruk Limau

## 📁 Project Structure

- `PrediksiJeruk_Final_Marco.ipynb` — Main notebook: preprocessing, model training, evaluation, and prediction.
- `dataset/` — Image data folder (must be structured as shown below).
- `model.h5` — Saved model file.
- `hasil_predict/` — Output prediction folder with visual results.

## 📊 Dataset Structure

dataset/
├── jeruk_mandarin/
├── jeruk_bali/
├── jeruk_lemon/
└── jeruk_limau/

markdown
Copy
Edit

Each subfolder contains images for its respective orange class.

## ⚙️ Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## 🚀 How to Run

1. Clone or download the repository.
2. Ensure the dataset is structured correctly (as shown above).
3. Open the Jupyter notebook: `PrediksiJeruk_Final_Marco.ipynb`.
4. Run all cells to:
   - Load and preprocess image data
   - Train and evaluate CNN model
   - Predict image classes
5. Save or load the trained model using:
   ```python
   model.save("model.h5")
   model = keras.models.load_model("model.h5")
✅ Model Evaluation
Evaluation metrics include accuracy, confusion matrix, and classification report.

The model shows good performance in recognizing the 4 types of oranges.

📌 Notes
You can experiment with additional layers, dropout rates, or optimizers to improve performance.

Consider applying data augmentation for better generalization.
