import json, sys, random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preparing dataset

train_dir = './datasets/training_data/training_data/images'
test_dir = './datasets/test_data_images/test_data_images/images'

batch_size=16

IGD = ImageDataGenerator(rescale=1./255,
                        horizontal_flip=True,
                        vertical_flip=True,
                        rotation_range=20,
                        validation_split=0.2)

train_generator = IGD.flow_from_directory(train_dir,
                                         target_size=(150,150),
                                         color_mode='rgb',
                                         batch_size=batch_size,
                                         shuffle=True,
                                         class_mode='categorical', 
                                         subset="training",
                                         seed=42)

val_generator = IGD.flow_from_directory(train_dir,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       color_mode='rgb',
                                       target_size=(150,150),
                                       class_mode='categorical', 
                                       subset="validation",
                                       seed=42)

# network design
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# optimization setup
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

# training
epochs = 100

result = model.fit(train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=epochs,
                   validation_data=val_generator,
                   validation_steps=val_generator.n//batch_size,
                   verbose=1)

# plot results
plt.style.use("ggplot")
fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[0].plot(result.history['loss'],label='Train Loss')
ax[0].plot(result.history['val_loss'],label='Validation Loss')
ax[1].plot(result.history['accuracy'],label='Train Accuracy')
ax[1].plot(result.history['val_accuracy'],label='Validation Accuracy')

ax[0].legend(loc='upper right')
ax[1].legend(loc='lower right')
plt.show();


# predictions
# pred = model.predict_generator(generator=val_generator)
# pred = np.argmax(val_pred,axis=1)

# class_name = list(val_generator.class_indices.keys())
# val_actual = val_generator.classes
# val_actual