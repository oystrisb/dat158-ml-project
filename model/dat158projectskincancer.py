import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set base directory
base_dir = '/kaggle/input/skin-cancer-malignant-vs-benign'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Define categories and labels
categories = {
    'malignant': 1,
    'benign': 0
}

# Function to load and preprocess images
def load_images_and_labels(base_dir, categories, img_size=(128, 128)):
    images, labels = [], []
    for category, label in categories.items():
        category_dir = os.path.join(base_dir, category)
        image_paths = glob(os.path.join(category_dir, '*.jpg'))
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB").resize(img_size)
            images.append(np.asarray(img) / 255.0)  # Normalize images
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess train/test data
train_images, train_labels = load_images_and_labels(train_dir, categories)
test_images, test_labels = load_images_and_labels(test_dir, categories)

# Display sample images
def display_sample_images(images, labels, title, n_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title('Malignant' if labels[i] == 1 else 'Benign')
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.show()

display_sample_images(train_images, train_labels, "Sample Training Images")

# Define CNN model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout for regularization
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Define ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Perform KFold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(train_images), 1):
    print(f"\nStarting Fold {fold}...\n")
    
    x_train, x_val = train_images[train_index], train_images[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]
    
    # Apply data augmentation on the training set
    train_generator = datagen.flow(x_train, y_train, batch_size=16)
    
    # Build the model
    model = build_model()
    
    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"fold_{fold}_model.weights.h5",
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=(x_val, y_val),
        epochs=15,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )
    
    # Evaluate on validation data
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=1)
    fold_accuracies.append(val_accuracy)
    print(f"Fold {fold} Validation Accuracy: {val_accuracy * 100:.2f}%")

# Final training with optimal epochs
mean_optimal_epochs = int(np.mean([early_stopping_cb.stopped_epoch for _ in range(3)]))
final_model = build_model()

final_model.fit(
    datagen.flow(train_images, train_labels, batch_size=16),
    validation_data=(test_images, test_labels),
    epochs=mean_optimal_epochs,
    batch_size=16
)

# Save the final model
final_model.save("final_skin_cancer_model.keras")

# Evaluate on test data
test_loss, test_accuracy = final_model.evaluate(test_images, test_labels, verbose=1)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# Visualize learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()

# Example learning curves for the last fold
plot_learning_curves(history)

# Predict on test images
predictions = final_model.predict(test_images[:5])
for i, prediction in enumerate(predictions):
    print(f"Image {i+1}: {'Malignant' if prediction > 0.5 else 'Benign'}")
