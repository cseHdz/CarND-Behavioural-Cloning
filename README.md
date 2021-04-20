# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/BGR2YUV.png "Conversion to YUV"
[image2]: ./output/flipped_Center.png "Flipped Image"
[image3]: ./output/left_recovery.png "Left Recovery Image"
[image4]: ./output/right_recovery.png "Right Recovery Image"
[image5]: ./output/Cropped.png "Cropped Image"
[image6]: ./output/normal.png "Normal Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* [videp.mp4](./output/video.mp4) Video of a lap in the first track in autonomous mode
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Three model architectures were implemented for purposes of this project. All models contain a Lambda normalization layer before any convolutions.

1. Model based on Lenet5 - This model is based on LeNet5, adjusted for a different input size with 3 channels. Two Dropout layers were added to reduce overfitting. (Lines 162 -189)
2. Model based on NVIDIA - The model on this paper below was implemented - https://arxiv.org/pdf/1604.07316v1.pdf  (Lines 129-160)
3. Custom Model - Simpple model with less than 15K parameters leveraged for testing of all other functions. (Lines 191-210)      

The final model used for this project was the one based on LeNet5. The model performed quite well after training on data from a lap on each track, and a lap in reverse on the first track.

The structure can be found below:

|Layer (type)           |     Output Shape            |   Param #    |
|:--------------------:|:------------------------------|-----------:| 
| lambda_1 (Lambda)            |(None, 70, 320, 3)    |    0         |
| conv2d_1 (Conv2D)            |(None, 66, 316, 6)    |    456       |
| max_pooling2d_1 (MaxPooling2 |(None, 33, 158, 6)    |    0         |
| conv2d_2 (Conv2D)            |(None, 29, 154, 16)   |    2416      |
| max_pooling2d_2 (MaxPooling2 |(None, 14, 77, 16)    |    0         |
| flatten_1 (Flatten)          |(None, 17248)         |    0         |
| dropout_1 (Dropout)          |(None, 17248)         |    0         |
| dense_1 (Dense)              |(None, 120)           |    2069880   |
| dense_2 (Dense)              |(None, 84)            |    10164     |
| dropout_2 (Dropout)          |(None, 84)            |    0         |
| dense_3 (Dense)              |(None, 1)             |    85        |
_________________________________________________________________

Total params: 2,083,001
Trainable params: 2,083,001
Non-trainable params: 0

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 179, 185). One uses a lower dropout of 30% and the second one of 50%. 
These were the same values I leveraged for my Traffic Sign Classification Model.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 229-252). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. It must be noted that throttle was adjusted during curves to ensure the car wasn't accelerating, which would likely set it off the drivable track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 235).
Batch size was set to 32 as I based model training on a relatively small dataset.

I leveraged an early stop callback in Keras to ensure training stopped if the network was no longer learning. The number of epics after which the network stopped was 15.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To better generalize the model, the following augmentation procedures were followed:

1. Center Driving: Randomly flip the image and respective angle (p-0.5).

![alt text][image6]
![alt text][image2]

2. Left/Right Recovery: The images from the two side cameras were only considered if they were associated with a minimal angle of 0.15 degrees. Then they were flipped randomly (p=0.5). In terms of the angle, both images were corrected with a constant of 0.24 (substract from right, add to left).

![alt text][image3]
![alt text][image4]

Given the amount of data, a batch generator was implemented to avoid loading all the data in memory. (Lines 60-124)

#### Details on training and testing data
After the collection process, I had 12,568 number of data points (incl. data from Udacity). I then preprocessed this data as follows:

1. Cropping the image: Since the top portion includes the horizon (70px) and the bottom is the nose of the vehicle (20px), those were removed. (Line 50)

![alt text][image5]

2. Adjusting the color space to YUV: The rationale was to follow the suggestion set out by the NVIDIA paper. (Line 51)

![alt text][image1]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
