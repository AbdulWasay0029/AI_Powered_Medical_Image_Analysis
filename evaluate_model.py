
# Step 5: Evaluate Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load trained model
model = load_model("medical_ai_model.h5")

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_dir = "chest_xray/test"
# Assuming 'chest_xray/test' directory exists and follows the same structure as 'train'
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Predict on test images (Make predictions for all samples)
y_true = test_data.classes
y_pred = model.predict(test_data)
y_pred = np.round(y_pred)

# Compute accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
