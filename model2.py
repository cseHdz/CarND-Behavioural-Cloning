# General Imports
import os
import math
import json
import csv
from pathlib import Path
import numpy as np

import cv2
from PIL import Image

# Keras Imports
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import Cropping2D
import keras.backend as K
from keras.utils import Sequence
from keras.regularizers import l2

# Sklearn
import sklearn
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split

# Load Data
def load_images(path, csv_file, correction=0.2):
    
    car_images = []
    steering_angles = []
    
    path = Path(path)
    
    with open(str(path / csv_file), 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip headers
        
        for row in reader:
             # create adjusted steering measurements for the side camera images
            correction = correction # this is a parameter to tune
            
            img1 = path / row[0]
            steering_center = float(row[3])
            
            if img1.is_file(): 
                car_images.append(str(img1))
                steering_angles.append(steering_center)
            
            img2 = path / row[1]
            steering_left = steering_center + correction
            
            if img2.is_file(): 
                car_images.append(str(img2))
                steering_angles.append(steering_left)
            
            img3 = path / row[2] 
            steering_right = steering_center - correction  
            
            if img3.is_file(): 
                car_images.append(str(img3))
                steering_angles.append(steering_right)    
    
    return car_images, steering_angles
    
#     images = []
#     angles = []
#     for i in range(0, len(car_images)):
        
#         img = np.asarray(Image.open(car_images[i]))
#         images.append(img)                 
#         images.append(cv2.flip(img,1))
        
#         angle = stering_angles[i]
#         angles.append(angle)
#         angles.append(angle * -1.0)
    
#     return images, angles


# Balance the data before training
def balance_data(car_images, measurements, upper_limit = 0.5, lower_limit = 0.03):
    
    m = np.array(measurements, dtype=np.float32)
    im = np.array(car_images, dtype=np.str)
    
    # Determine the distribution of data
    # Bin the data every 0.2 degrees
    min_angle = min(measurements)
    max_angle = max(measurements)
    num_bins = math.ceil((max_angle-min_angle)/0.1)
    
    bins = [min_angle + 0.1 * b for b in range(0, num_bins)]
    hist = np.histogram(measurements, bins=bins)     
    
    # Calculate proportions
    num_samples = len(measurements)
    proportion = hist[0]/num_samples
    
    
    p3 = math.ceil(num_samples * lower_limit)
    p30 = math.ceil(num_samples * upper_limit)
    
    car_images_balanced = []
    measurements_balanced = []
    
    for i in range(0, len(hist[1]) - 1):
        
        subset_y = m[(m >= hist[1][i]) &  (m < hist[1][i + 1])]
        subset_x = im[(m >= hist[1][i]) &  (m < hist[1][i + 1])]
        
        if proportion[i] > upper_limit:
            # No class should represent more than 30% if the data. Downsample.
             x, y = resample(subset_x, subset_y, n_samples=p30, random_state=42)
            
        elif proportion[i] < lower_limit:
            # Make sure minority classes represent at least 3% of the original data. Upsample.
            x, y = resample(subset_x, subset_y, n_samples=p3, random_state=42)
        
        else:
            x, y = subset_x, subset_y
        
        car_images_balanced.extend(x)
        measurements_balanced.extend(y)
        
    return car_images_balanced, measurements_balanced
   
    
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
        
        images = []
        angles = []
        for i in range(0, len(X_batch)):

            image = np.asarray(Image.open(X_batch[i]))
            angle = float(y_batch[i])
            
            images.append(cv2.flip(image,1))
            angles.append(angle*-1.0)

            images.append(image)
            angles.append(angle)

        X_train = np.array(images)
        y_train = np.array(angles)
        
        return shuffle(X_train, y_train)
    
    def on_epoch_end(self):
        
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)   
   



# Load Data
car_images = []
measurements = []

folders = ['/opt/carnd_p3/data/', '/opt/simulations', '/opt/simulations2']

for folder in folders:
    im, angles = load_images(folder, 'driving_log.csv')
    car_images.extend(im)
    measurements.extend(angles)
    
# car_images, measurements = balance_data(car_images, measurements, 
#                                         upper_limit = 0.5, lower_limit = 0.00)

# # Split data between training and testing
X_train, X_test, y_train, y_test  = train_test_split(car_images, measurements, test_size=0.2)

# Building the Model
model = build_NVIDA_model()

print(model.summary())

# Training & Saving
epochs = 5
batch_size = 32

training_data = BatchGenerator(X_train, y_train, batch_size=batch_size, shuffle=True) 
validation_data = BatchGenerator(X_test, y_test, batch_size=batch_size, shuffle=True) 

model.fit_generator(generator=training_data,
                    validation_data=validation_data,
                    epochs=epochs,
                    steps_per_epoch=int(math.ceil(len(X_train)/batch_size)),
                    validation_steps=int(math.ceil(len(X_test)/batch_size)))

# model.fit(car_images,measurements,validation_split=0.2,shuffle=True,epochs=epochs, batch_size=batch_size)                
model.save('./model.h5')
    
# # 6. Save Model/Weights
# with open('model.json', 'w') as outfile:
#     json.dump(model.to_json(), outfile)
# model.save_weights('model.h5')