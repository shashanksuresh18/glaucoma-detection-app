import pandas as pd
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# --- Step 1: Data Preparation ---
print("--- Starting Step 1: Data Preparation ---")
labels_path = 'data/Labels.csv'
df = pd.read_csv(labels_path, encoding='utf-8-sig')

id_col = 'Image Name'
label_col = 'Label'
df[label_col] = df[label_col].astype(str)

image_folder = 'data/Images/'
df['filepath'] = df[id_col].apply(lambda x: os.path.join(image_folder, x))

train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df[label_col])
train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42, stratify=train_val_df[label_col])

print("Data preparation complete. ✅\n")


# --- Step 2: Data Augmentation and Loading ---
print("--- Starting Step 2: Setting up Data Generators ---")
IMG_SIZE = (299, 299) # InceptionV3 requires 299x299 images
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col=label_col,
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col=label_col,
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False
)
print("Data generators are ready. ✅\n")


# --- Step 3: Build the Transfer Learning Model ---
print("--- Starting Step 3: Building the Model ---")
base_model = InceptionV3(
    # THE FIX IS HERE: Changed 'IMG_size' to 'IMG_SIZE'
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Model built successfully. ✅\n")
model.summary()


# --- Step 4: Train the Model ---
print("\n--- Starting Step 4: Training the Model ---")
EPOCHS = 15

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)
print("\nTraining complete! 🎉")

import matplotlib.pyplot as plt

# --- Step 5: Visualize Training History ---
print("\n--- Starting Step 5: Plotting Training History ---")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

print("Plotting complete. ✅\n")

# --- Step 6: Evaluate on the Test Set ---
print("--- Starting Step 6: Evaluating on the Test Set ---")

# Create a generator for the test data (don't augment it)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col=label_col,
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False  # Important: Do not shuffle test data
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

print("\nModel evaluation complete. ✅")

# --- Step 7: Save the Model ---
print("\n--- Starting Step 7: Saving the Model ---")
model.save('glaucoma_detector_model.h5')
print("Model saved as 'glaucoma_detector_model.h5' ✅")