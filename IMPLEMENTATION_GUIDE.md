
# 🏥 How to Build the "AI-Powered Medical Image Analysis" Project From Scratch

This guide documents the exact steps taken to build this project, following the original PDF requirements. You can use this as a tutorial to recreate the project yourself.

## Phase 1: Preparation (The Setup)

1.  **Install Python**:
    *   Download and install **Python 3.11** (recommended for TensorFlow compatibility).
    *   During installation, check "Add Python to PATH".

2.  **Create Project Folder**:
    *   Create a folder named `Medical_Image_Analysis`.
    *   Open this folder in VS Code.

3.  **Get the Data**:
    *   Go to Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
    *   Download the dataset (approx 1-2 GB).
    *   Extract the ZIP file inside your project folder.
    *   **Important**: Ensure your folder structure looks like this:
        ```
        Medical_Image_Analysis/
        └── chest_xray/
            ├── train/
            ├── test/
            └── val/
        ```

4.  **Set Up Virtual Environment**:
    *   Open VS Code Terminal (`Ctrl+``).
    *   Run: `py -3.11 -m venv env`
    *   Activate: `.\env\Scripts\activate`

5.  **Install Libraries**:
    *   Create a file `requirements.txt` with:
        ```text
        tensorflow
        opencv-python
        matplotlib
        scikit-learn
        flask
        numpy
        ```
    *   Run: `pip install -r requirements.txt`
    *   *Troubleshooting*: If you get "File path too long" errors on Windows, map a virtual drive:
        ```powershell
        subst T: "C:\path\to\your\Medical_Image_Analysis"
        T:
        pip install tensorflow
        ```

---

## Phase 2: Building the AI (The Code)

### Step 1: See the Data (`load_preview.py`)
Create a script to make sure regular Python can read the X-rays.
*   **Goal**: Open an image from `chest_xray/train/PNEUMONIA` and show it.
*   **Key Code**: `cv2.imread(path)` and `plt.imshow()`.

### Step 2: Train the Brain (`train_model.py`)
This is the core AI part (Step 4 in your guide).
1.  **Preprocessing**: Use `ImageDataGenerator` to rescale pixels (1/255) and augment data (zoom/rotate) to make the AI smarter.
2.  **Model Building**: Create a `Sequential` CNN model.
    *   `Conv2D`: The "eyes" that detect edges/shapes.
    *   `MaxPooling2D`: Reduces size to focus on important features.
    *   `Flatten`: Converts 2D image features to a 1D list.
    *   `Dense`: The "brain" that makes the final decision.
3.  **Training**: Run `model.fit()` for 10 epochs.
4.  **Saving**: Save the result as `medical_ai_model.h5`.

### Step 3: Test the Brain (`evaluate_model.py`)
Check if the AI is actually learning (Step 5 in your guide).
*   **Goal**: Load `medical_ai_model.h5` and predict on the `test` folder.
*   **Metric**: Calculate Accuracy % and show a Confusion Matrix (True Positives vs False Positives).

---

## Phase 3: Building the App (The Interface)

### Step 4: Create the Web Server (`app.py`)
Use Flask to make the AI accessible (Step 6 in your guide).
1.  **Load Model**: Load `medical_ai_model.h5` when the app starts.
2.  **Define Routes**:
    *   `/` (Home): Serve the HTML page.
    *   `/predict` (API): Accept an image via POST, preprocess it (resize to 256x256), and return "Pneumonia" or "Normal".

### Step 5: Create the Frontend (`templates/index.html`)
Build a simple UI.
*   **Elements**: File input, "Analyze" button, and Result display.
*   **JavaScript**: Send the image to `/predict` and show the result without reloading the page.

---

## Phase 4: Usage

1.  **Train**: `python train_model.py` (Wait ~15 mins).
2.  **Run**: `python app.py`.
3.  **Use**: Open browser at `http://127.0.0.1:5000`.

## ⚠️ Common Pitfalls to Avoid
1.  **Wrong Python Version**: TensorFlow 2.10+ works best with Python 3.10-3.11. Avoid 3.12+ for now.
2.  **Files Not Found**: Always check if your `chest_xray` folder is in the right place relative to your scripts.
3.  **Windows Path Limits**: TensorFlow installs deep folder structures. Use short paths or `subst` drives to avoid crashes.
