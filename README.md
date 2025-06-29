🍊 Orange Image Classification using CNN
This project is a deep learning-based image classification system designed to identify four types of oranges using Convolutional Neural Networks (CNNs). It was developed as part of a computer vision coursework assignment using TensorFlow and Keras.

📁 Project Structure
PrediksiJeruk_Final_Marco.ipynb: Main Jupyter Notebook containing data preprocessing, model training, evaluation, and prediction.

dataset/: Folder containing training, validation, and test images categorized into 4 types of oranges:

jeruk_mandarin/

jeruk_bali/

jeruk_lemon/

jeruk_limau/

model.h5: Saved model file after training.

hasil_predict/: Output folder for prediction results with visualizations.

🚀 Features
Custom CNN architecture using Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.

Image preprocessing including resizing and normalization.

Multi-class classification for 4 orange types.

Real-time prediction support on unseen images.

Model evaluation using confusion matrix and classification report.

🛠️ Technologies Used
Python 3

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

🧪 How to Use
Prepare the dataset directory with the following structure:

Copy
Edit
dataset/
  ├── jeruk_mandarin/
  ├── jeruk_bali/
  ├── jeruk_lemon/
  └── jeruk_limau/
Open the PrediksiJeruk_Final_Marco.ipynb notebook.

Run each cell to:

Load and preprocess the dataset.

Train the CNN model.

Evaluate the model.

Predict and visualize new test images.

Save or load the model using:

python
Copy
Edit
model.save("model.h5")
model = keras.models.load_model("model.h5")
📊 Evaluation
The CNN model successfully classifies 4 types of orange images.

Evaluation includes a confusion matrix and a detailed classification report (precision, recall, F1-score).

📌 Notes
Ensure that the dataset folder is present and structured correctly.

You can enhance performance by using data augmentation or experimenting with different CNN architectures.
