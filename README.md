
# AI-Powered Medical Image Analysis - Pneumonia Detection

This project implements an AI system using a **Convolutional Neural Network (CNN)** to detect **Pneumonia** from **Chest X-Ray images**.
It follows the step-by-step implementation guide:
1.  **Data Loading**: Automatically loads and previews X-ray images.
2.  **Preprocessing**: Resizes (256x256), converts to grayscale, and augments data.
3.  **Model Training**: Trains a CNN (Conv2D -> MaxPooling -> Dense) on the dataset.
4.  **Evaluation**: Checks accuracy and generates a confusion matrix.
5.  **Web Deployment**: Provides a Flask web interface for users to upload and analyze images.

## 🛠️ Project Structure
```
Medical_Image_Analysis/
├── app.py               # Flask Web Application (Step 6)
├── train_model.py       # Training Script (Steps 3 & 4)
├── evaluate_model.py    # Evaluation Script (Step 5)
├── load_preview.py      # Data Loading Script (Step 2)
├── medical_ai_model.h5  # Trained AI Model
├── requirements.txt     # Python Dependencies
├── templates/
│   └── index.html       # Web Interface (Frontend)
└── chest_xray/          # Dataset (Train/Test/Val)
```

## 🚀 How to Recreate This Project
Follow these exact steps to run the project from scratch.

### 1. Prerequisites
- **Python 3.10 or 3.11** (Note: TensorFlow 2.x is not yet compatible with Python 3.12+ as of early 2026).
- **Dataset**: Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

### 2. Setup Environment
Open your terminal in the project folder:
```bash
# Create a virtual environment (Python 3.11 recommended)
py -3.11 -m venv env

# Activate the environment
# Windows:
.\env\Scripts\activate
# Mac/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```
*(Note for Windows users: If you encounter path length errors during installation, use a virtual drive mapping `subst T: "path/to/project"` and install on `T:`)*

### 3. Run the Scripts (In Order)

**Step 1: Preview Data**
Verify the dataset is loaded correctly.
```bash
python load_preview.py
```
*Output: Saves a sample X-ray image as `preview_sample.png`.*

**Step 2: Train the AI Model**
Train the CNN model on your dataset. This may take 10-20 minutes on CPU.
```bash
python train_model.py
```
*Output: Saves the trained model as `medical_ai_model.h5`.*

**Step 3: Evaluate Performance**
Check how accurate the model is on unseen test data.
```bash
python evaluate_model.py
```
*Output: Prints accuracy (e.g., 76%) and a confusion matrix.*

**Step 4: Launch the Web App**
Start the Flask server to use the AI interactively.
```bash
python app.py
```
*   Open your browser and visit: **http://127.0.0.1:5000**
*   Upload an X-ray image (e.g., from `chest_xray/test/PNEUMONIA`) to see the result.

## 📊 Model Architecture
- **Input**: 256x256 Grayscale Image
- **Layers**:
  - Conv2D (32 filters) + MaxPooling
  - Conv2D (64 filters) + MaxPooling
  - Flatten
  - Dense (128 neurons, ReLU) + Dropout (0.5)
  - Output Dense (1 neuron, Sigmoid) for Binary Classification (Normal vs Pneumonia).
