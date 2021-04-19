
import os
import json
import csv
from pathlib import Path

import numpy as np

# Image Modification
import cv2
from PIL import Image
from scipy import misc

from keras.callbacks import EarlyStopping
# Keras Imports

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
# from keras.layers.pooling import AveragePooling2D, MaxPooling2D
# from keras.layers import Cropping2D

# import keras.backend as K
from keras.utils import Sequence
# from keras.regularizers import l2

# Sklearn
import sklearn
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split

# Load Images
def load_images(file_path, file):

    with open(Path(file_path) / file, 'r') as f:
        reader = csv.reader(f)
        
        next(reader) # Skip header
        X = [row[0:2] for row in reader]
        y = [row[3] for row in reader]
    
    return X, y

def preprocess_image(image):
    
    img = image[70:140,:]
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2YUV))[:,:,1]
    return misc.imresize(img, (18,80,1))


#######################
# Generate Batches
######################

class BatchGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle):
        self.x, self.y = np.array(x_set), np.array(y_set)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        
        X_batch = self.x[indexes]
        y_batch = self.y[indexes]
    
        return self.__get_images(X_batch, y_batch)
        
    def __get_images(self, X_batch, y_batch):
        
        X = []
        y = []
        
        def augment(img_path, angle):
            
            # Preprocess the image
            image = np.asarray(preprocess_image(Image.open(img_path)))
            
            # Flip images randomly
            if np.random.choice(2) == 1:       
                X.append(cv2.flip(image,1))
                y.append(angle*-1.0)
            else:
                X.append(image)
                y.append(angle)
        
        for i in range(0, len(X_batch)):

            angle = float(y_batch[i])
            
            # Add the center image
            augment(X_batch[i][0], angle)
            
            # Add left and right cameras when above a min. threshold
            if abs(angle) > 0.15:
                augment(X_batch[i][1], angle + correction)
                augment(X_batch[i][2], angle - correction)
                            
        X_train = np.array(X)
        X_train= np.reshape(X_train, (-1, 18,80,1))
        
        y_train = np.array(y)
        
        return X_train, y_train
    
    def on_epoch_end(self):
        
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)   
   
#########################
# NVIDIA Model
#########################
def build_NVIDA_model():
    """
        Basic implementation of the NVIDIA Self-Driving Vehicle
        model suggested in project instructions.
        
        Source: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
        Paper: https://arxiv.org/pdf/1604.07316v1.pdf        
    
    """
    
    model = Sequential()
    
    model.add(Lambda(lambda x: x /255.0 - 1.0, input_shape=(160,320,3)))
     
    # 2-Stride Convolutional Layers
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
    
    # Non-Strided Convolutional Layer
    model.add(Conv2D(filters=48, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    
    model.add(Flatten())
    
    # Dense Layers
    model.add(Dense(units=1164, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='relu'))
    
    model.compile(optimizer='adam', loss = 'mse', metrics=['mean_squared_error'])
    
    return model              


###################################
#
###################################

X, y = load_images('/opt/data', 'driving_log.csv')
              
# Split data between training and testing
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Model
model = build_NVIDA_model()
print(model.summary())

# Training & Saving
epochs = 100
batch_size = 32

early_stop= EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, 
                          patience=50, verbose=1, mode='min')

training_data = BatchGenerator(X_train, y_train, batch_size=batch_size, shuffle=True) 
validation_data = BatchGenerator(X_test, y_test, batch_size=batch_size, shuffle=True) 

model.fit_generator(generator=training_data,
                    validation_data=validation_data,
                    epochs=epochs, callbacks = [early_stop],
                    steps_per_epoch=int(math.ceil(len(X_train)/batch_size)),
                    validation_steps=int(math.ceil(len(X_test)/batch_size)))
              
model.save('./model.h5')
   