# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set image and batch size
IMAGE_SIZE = 256
BATCH_SIZE = 32

# Set the directory paths for the training and validation sets
train_dir = 'C:/Users/JACK JMM/Desktop/notebook/potato leaf disease predictor/train'
val_dir = 'C:/Users/JACK JMM/Desktop/notebook/potato leaf disease predictor/validation'

# Define the data generators for the training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')

# Define the input shape and number of classes
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
num_classes = 3

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 10

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the model
model.save('potato_disease_model.h5')

