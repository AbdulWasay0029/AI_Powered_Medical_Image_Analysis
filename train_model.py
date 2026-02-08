
# Step 3: Preprocessing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("TensorFlow version:", tf.__version__)

# Define Image Data Generator for Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2 # Added validation split for better training monitoring
)

# Load dataset
# Ensure the dataset folder 'chest_xray' is present in the current directory
train_dir = "chest_xray/train"
# If using Kaggle dataset structure, there might be 'train', 'test', 'val' folders.
# If only 'train' exists, we use validation_split.

# Training Data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset='training'
)

# Validation Data (if splitting from train directory)
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset='validation'
)

# Step 4: Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Binary classification: Normal vs Pneumonia
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
print("Starting training...")
# Step 4 snippet says epochs=10.
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the trained model
model.save("medical_ai_model.h5")
print("Model saved as medical_ai_model.h5")
