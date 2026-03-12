# AI Medical Image Analysis — Pneumonia Detection

CNN-based binary classification system for detecting pneumonia from chest X-ray images. Trained on the Kaggle Chest X-Ray dataset, deployed via a Flask web interface for real-time inference.

**Accuracy: ~76% on test set**

---

## Model Architecture

```
Input (256×256 grayscale)
  → Conv2D (32 filters, ReLU) → MaxPooling
  → Conv2D (64 filters, ReLU) → MaxPooling
  → Flatten
  → Dense (128, ReLU) → Dropout (0.5)
  → Dense (1, Sigmoid)        → Binary output: Normal / Pneumonia
```

---

## Stack

**Python · TensorFlow · Keras · Flask · NumPy · OpenCV**

---

## Project Structure

```
├── app.py                # Flask web application
├── train_model.py        # Model training script
├── evaluate_model.py     # Evaluation + confusion matrix
├── load_preview.py       # Dataset preview utility
├── medical_ai_model.h5   # Trained model weights
├── requirements.txt
└── templates/
    └── index.html        # Upload interface
```

---

## Setup

**Requirements:** Python 3.10 or 3.11 (TensorFlow 2.x is not compatible with Python 3.12+)

```bash
# Create virtual environment
py -3.11 -m venv env

# Activate (Windows)
.\env\Scripts\activate

# Activate (Mac/Linux)
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Windows note:** If you hit path length errors during pip install, map a virtual drive:
> `subst T: "C:\path\to\project"` then install from `T:`

---

## Dataset

Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle. Extract into `chest_xray/` at the project root. The folder should contain `train/`, `test/`, and `val/` subdirectories.

---

## Usage

Run scripts in this order:

```bash
# 1. Verify dataset loads correctly
python load_preview.py
# Output: preview_sample.png

# 2. Train the model (10–20 min on CPU)
python train_model.py
# Output: medical_ai_model.h5

# 3. Evaluate on test set
python evaluate_model.py
# Output: accuracy score + confusion matrix

# 4. Launch web interface
python app.py
# Open: http://127.0.0.1:5000
```

Upload any chest X-ray image through the browser interface to get a Normal / Pneumonia prediction.

---

## Preprocessing Pipeline

- Resize to 256×256
- Convert to grayscale
- Normalize pixel values to [0, 1]
- Data augmentation on training set (rotation, horizontal flip)
