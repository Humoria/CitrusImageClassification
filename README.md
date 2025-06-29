🍊 **JERUKNET**  
**JerukNet** is a deep learning-powered image classification system built to automatically recognize four distinct types of oranges: **Jeruk Mandarin, Jeruk Bali, Jeruk Lemon, and Jeruk Limau**. This application leverages a convolutional neural network (CNN) model for multi-class image classification tasks.

---

### 🚀 Features

✅ Classifies orange images into 4 types  
🧠 Built with CNN architecture using TensorFlow/Keras  
📷 Supports custom dataset loading and preprocessing  
📊 Visual performance reports (accuracy & confusion matrix)  
🧪 Model prediction support for new, unseen images  
💾 Trained model can be saved and reused (`.h5` format)  

---

### 🛠️ Tech Stack

- **Machine Learning**: TensorFlow, Keras  
- **Language**: Python 3  
- **Notebook**: Jupyter Notebook  
- **Visualization**: Matplotlib  
- **Evaluation**: Scikit-learn  

---

### 📁 Project Structure

orange-classifier/
├── PrediksiJeruk_Final_Marco.ipynb # Main notebook (training, prediction)
├── dataset/
│ ├── jeruk_mandarin/
│ ├── jeruk_bali/
│ ├── jeruk_lemon/
│ └── jeruk_limau/
├── model.h5 # Saved CNN model
├── hasil_predict/ # Folder for prediction results
└── README.md

yaml
Copy
Edit

---

### 📸 How It Works

1. Prepare a dataset with 4 folders (each for a type of orange).
2. Run the notebook `PrediksiJeruk_Final_Marco.ipynb`.
3. CNN model is trained on the dataset.
4. Evaluation results are displayed (confusion matrix, metrics).
5. New images can be classified using the trained model.

---

### 🧠 Model Details

The model uses a **custom CNN architecture** composed of:
- Convolutional Layers with ReLU activation  
- Max Pooling for dimensionality reduction  
- Fully Connected (Dense) Layers  
- Dropout for regularization

You can fine-tune, retrain, or expand the model to classify additional fruit types if desired.

---

### 🧪 Use Case

This project is suitable for:
- Agricultural classification systems  
- Smart farming and inventory automation  
- Educational tools for fruit recognition  
- Computer vision training tasks  

---

### ⚙️ Setup & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/orange-classifier.git
   cd orange-classifier
Install required packages:

bash
Copy
Edit
pip install tensorflow keras matplotlib scikit-learn
Run the notebook:

bash
Copy
Edit
jupyter notebook PrediksiJeruk_Final_Marco.ipynb
📦 Dependencies
TensorFlow

Keras

Matplotlib

NumPy

Scikit-learn

👨‍💻 Author
Created with 🍊 by Marco Albert
For academic and experimental use only.
