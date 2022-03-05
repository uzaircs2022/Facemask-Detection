# Facemask-Detection

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import cv2
import numpy as np
import os
import csv

trainingDataset = []
img_size = 100
path = "/content/drive/My Drive/Pictures/Train"

classNumber = 0
trainingDataset.clear()

for folder in (os.listdir(path)):
  print(classNumber)
  print("Folder Name:",folder)
  # folder = with_mask ,without_mask
  fp = os.path.join(path,folder)
  # joining folder like /content/Face_Mask/Train/with_mask
  for eachImage in os.listdir(fp):
    imagePath = os.path.join(fp,eachImage)
    img = (cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE))/255
    resize=cv2.resize(img,(img_size,img_size))
    trainingDataset.append([resize,classNumber])
  classNumber = classNumber + 1
  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import pickle
import time

X = []
Y = []
img_size = 100
np.random.shuffle(trainingDataset)
for features, label in trainingDataset:
    X.append(features)
    Y.append(label)
print(Y) 

X = np.array(X).reshape(-1, img_size, img_size, 1)
Y_binary = to_categorical(Y)
print(Y_binary)

model = Sequential()

model.add(Conv2D(200, (3, 3), input_shape=(100,100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
 
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
              
             
model.fit(X, Y_binary,
          batch_size = 32,
          epochs=20, validation_split = 0.1)
 
model.save("/content/drive/My Drive/face_mask/Models/{NAME}.model")

def prepare(filepath):
    img_size = 100 
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255  
    img_resize = cv2.resize(img, (img_size, img_size))  
    return img_resize.reshape(-1, img_size, img_size, 1)
    
    
prediction = model.predict(prepare("/content/drive/My Drive/Pictures/Test/With Mask/check.jpg"))
print((prediction))

CATEGORIES = ["with_mask", "without_mask"]

pred_class = CATEGORIES[np.argmax(prediction)]
print(pred_class)

