import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from cvzone.FaceDetectionModule import FaceDetector

path = "TNK-M10-Pneumothorax-New-Dataset-main"

images = []
categories = []

for img in os.listdir(path):
    try:
        print(img)
    
        type = img.split("_")[0]
        img = cv2.imread(str(path)+"/"+str(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(200,200))
        images.append(img)
        categories.append(type)
               
    except:
        print("error in reading")

print("Count of all images", len(images))

# Change the categories to numpy array of int64
categories = np.array(categories,dtype=np.int64)
# Change images to numpy array
images = np.array(images)

# Split the images and categories using train_test_split
training_images, testing_images, training_categories, testing_categories = train_test_split(images, categories)

# Create a sequential model
model = Sequential()

# Add First layer of the model
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
model.add(MaxPool2D(pool_size=3, strides=2))
# Add Second layer of the model
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))
# Add thirds layer of the model              
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))
# Add fourth layer of the model
model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))
# Add flatten to model
model.add(Flatten())
# Add Dropout of 0.2 to model
model.add(Dropout(0.2))
# Add dense layer to model with 512 size and relu activation
model.add(Dense(512, activation='relu'))
# Add dense layer to model with 1 size and linear activation and name = 'age'
model.add(Dense(1, activation='linear', name='categories'))
# Compile the model              
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Print model summary
print(model.summary())
# tarin the model
history = model.fit(training_images, training_categories,
validation_data=(testing_images, testing_categories), epochs=10)
# Save the model
model.save('model_10epochs.h5')
# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

acc = history.history['accuracy']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()