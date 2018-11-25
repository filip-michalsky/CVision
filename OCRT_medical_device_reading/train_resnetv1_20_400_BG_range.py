'''
This script for training CNN networks specifically for medical device digits recognition
is under commercial license for CheckMate Diabetes, Inc. and has been released
for EDUCATIONAL PURPOSES ONLY.
'''

import os
import sys
import time
import re
import datetime as dt
import imageio
import gc
import warnings
import time


import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from keras import backend as K
from keras.models import Model


import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from io import BytesIO
from tqdm import tqdm
from PIL import Image

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

sns.set_palette('muted',color_codes=True)
warnings.filterwarnings("ignore")

def get_list_of_image_names(direct):
    '''
    Returns a list of image names in a given directory
    '''
    candidate_files = os.listdir(direct)
    images = []
    for file in candidate_files:
        if file[-3:].lower() == 'jpg':
            images.extend([file])
    return images

def load_images(dir_list,obs,height=128,width=128, check_imgs_flag=False):
    '''
    Load the existing images into Jupyter Notebook
    and assign labels, then randomly shuffle the data and finally split into train
    and test sets.
    
    test_size: size of the test set as % of overall data
    '''
    
    labels = []
    total_cnt = 0
    check_imgs = []

    # pre-allocate numpy array for img data
    X_data = np.zeros((obs,height,width,3),dtype='float32')
    
    for curr_direc in tqdm(dir_list):
        
        image_names = get_list_of_image_names(curr_direc)
        
        cnt = 0 # start incrementing the count of directories
        labels_cnt = 0
        for image in image_names:
            # load each image one at a time
            loaded_image = imageio.imread(curr_direc+'/'+image)
            
#             if cnt%2==0: # only do odd number of images to decrease dataset size
            # cast to a numpy array
            if check_imgs_flag:
                check_imgs.append(loaded_image)
            converted_image = np.array(loaded_image,dtype='float32')

            X_data[total_cnt]= converted_image
            total_cnt+=1
            labels_cnt+=1
            cnt+=1
            #gc.collect()
        
        # beginning of the directory name
        labels.extend([curr_direc[14:]]*labels_cnt)
    y_data = np.array(labels,dtype='int32')

    if check_imgs_flag:
        return X_data,y_data, check_imgs
    else:
        return X_data,y_data
        

#### IMAGE PREPROCESSING #######

# add the prefix 'results' to our list of dir names
dir_list_full = ["data_imgs_all/"+str(i) for i in range(20,400)]


obs_cnt =0
for curr_direc in tqdm(dir_list_full):
    image_names = get_list_of_image_names(curr_direc)
    curr_count = 0
    for image in image_names:
            # load each image one at a time
            obs_cnt+=1
            curr_count+=1
print(obs_cnt)

X_data, y_data = load_images(dir_list=dir_list_full,obs=obs_cnt)

# add the prefix 'results' to our list of dir names
alarm_nums = [str(i) for i in range(100,160)]+ [str(i) for i in range(200,260)]+ [str(i) for i in range(300,360)] + [str(i) for i in range(20,60)]
dir_list_full = ["alarm_dig_min/"+str(i) for i in alarm_nums]


obs_cnt =0
for curr_direc in tqdm(dir_list_full):
    image_names = get_list_of_image_names(curr_direc)
    curr_count = 0
    for image in image_names:
            # load each image one at a time
            obs_cnt+=1
            curr_count+=1
    #obs_cnt+=len(image_names)
print(obs_cnt)

X_data_alarm, y_data_alarm = load_images(dir_list=dir_list_full,obs=obs_cnt)

X_data_complete = np.vstack((X_data,X_data_alarm))

del X_data
del X_data_alarm
gc.collect()

y_data_complete = np.append(y_data,y_data_alarm)

def create_train_test_data(class_data, labels, test_size=0.2, preprocess = False,standardize=False):
    '''
    preprocess: flag whether to perform preprocessing
    standardize: flag whether to perform standardization
    '''
    # now create numpy arrays and shuffle data
    
    print('Shuffling...')
    idx = np.array(range(class_data.shape[0]),dtype=int)
    np.random.shuffle(idx)
    
    print('Assigning shuffled indices...')
    data = class_data[idx]
    labels = labels[idx]
    gc.collect()
    # get test_set indices
    
    print('Splitting array to train/test...')
    X_test,X_train = np.split(data,[np.int(np.floor(len(idx)*test_size))])
    if preprocess:
        print('standardizing colors..')
        X_test_post = preprocess_img(X_test,standardize=standardize)
        del(X_test)
        gc.collect()
        X_train_post = preprocess_img(X_train,standardize=standardize)
        del(X_train)
        gc.collect()
        y_test,y_train = np.split(labels,[np.int(np.floor(len(idx)*test_size))])
        return X_train_post,y_train,X_test_post,y_test
    
    y_test,y_train = np.split(labels,[np.int(np.floor(len(idx)*test_size))])
    
    return X_train,y_train,X_test,y_test

def preprocess_img(X_train,standardize=True):
    '''
    Preprocess each of RGB layers in the dataset
    If standardize, then divide by standard deviation.
    Otherwise, just subtract mean of dataset
    '''
    #X_train_std = X_train.copy()

    for idx,img in enumerate(X_train):
        
        r_mean = np.mean(img[:,:,0],dtype=float) 
        b_mean = np.mean(img[:,:,1],dtype=float) 
        g_mean = np.mean(img[:,:,2],dtype=float) 

        r_std = np.std(img[:,:,0],dtype=float) 
        b_std = np.std(img[:,:,1],dtype=float) 
        g_std = np.std(img[:,:,2],dtype=float) 
        #print(img[:,:,0].shape,r_mean.shape,r_std.shape)
        if standardize:
            X_train[idx][:,:,0] = (img[:,:,0]-r_mean)/r_std
            X_train[idx][:,:,1] = (img[:,:,1]-b_mean)/b_std
            X_train[idx][:,:,2] = (img[:,:,2]-g_mean)/g_std
        else:
            X_train[idx][:,:,0] = (img[:,:,0]-r_mean)
            X_train[idx][:,:,1] = (img[:,:,1]-b_mean)
            X_train[idx][:,:,2] = (img[:,:,2]-g_mean)

    return X_train
    
x_train,y_train,x_test,y_test = create_train_test_data(X_data_complete,y_data_complete,preprocess=False)


def prepare_labels_num_digits_recognition(y_train):
    # Convert labels into indicator whether image has two or three digits
    new_labels = np.zeros((len(y_train)),dtype='int32')
    for idx,i in enumerate(y_train):
        new_labels[idx] = len(list(str(i)))
    return new_labels

def reassign_digit_label(y_train,digit_desired):
    '''
    Reassign labels to correspond to desired learning outcome of NN.
    
    digit_desired: first, second, or third digit to recognize : select 0, 1, or 2 
    '''
    
    new_labels = np.zeros((len(y_train)),dtype='int32')
    
    # now loop through relevant labels and find what number is at out desired position
    for idx,i in enumerate(y_train):
        i = str(i)
        if len(i)==2:
            if digit_desired ==0: # first digit (the highest) can be one of 11 categories
                new_labels[idx] = 0
            elif digit_desired ==1: # second digit can be one of 10 categories
                new_labels[idx] = int(list(str(i))[0])
            elif digit_desired ==2:
                new_labels[idx] = int(list(str(i))[1])
        elif len(i) ==3:
            if digit_desired ==0: # first digit can be one of 10 categories
                new_labels[idx] = int(list(str(i))[0])
            elif digit_desired ==1: # second digit can be one of 10 categories
                new_labels[idx] = int(list(str(i))[1])
            elif digit_desired ==2: # second digit can be one of 10 categories
                new_labels[idx] = int(list(str(i))[2])
                
    return new_labels
            

# convert labels for train set
length_labels_train = prepare_labels_num_digits_recognition(y_train)
first_digit_train = reassign_digit_label(y_train,digit_desired = 0)
second_digit_train = reassign_digit_label(y_train,digit_desired = 1)
third_digit_train = reassign_digit_label(y_train,digit_desired = 2)

# convert labels for test set
length_labels_test = prepare_labels_num_digits_recognition(y_test)
first_digit_test = reassign_digit_label(y_test,digit_desired = 0)
second_digit_test = reassign_digit_label(y_test,digit_desired = 1)
third_digit_test = reassign_digit_label(y_test,digit_desired = 2)

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
# We need a list of 4, in order of length, digit 1, digit 2, digit 3

length_train = keras.utils.to_categorical(length_labels_train - 2, len(np.unique(length_labels_train)))
length_test = keras.utils.to_categorical(length_labels_test - 2, len(np.unique(length_labels_test)))

digit1_train = keras.utils.to_categorical(first_digit_train, len(np.unique(first_digit_train)))
digit1_test = keras.utils.to_categorical(first_digit_test, len(np.unique(first_digit_test)))

digit2_train = keras.utils.to_categorical(second_digit_train, len(np.unique(second_digit_train)))
digit2_test = keras.utils.to_categorical(second_digit_test, len(np.unique(second_digit_test)))

digit3_train = keras.utils.to_categorical(third_digit_train, len(np.unique(third_digit_train)))
digit3_test = keras.utils.to_categorical(third_digit_test, len(np.unique(third_digit_test)))

y_train_enc = [length_train, digit1_train, digit2_train, digit3_train]
y_test_enc = [length_test, digit1_test, digit2_test, digit3_test]



################################################################################################################################################################################################################################################

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)


# Input image dimensions.
input_shape = x_train.shape[1:]


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 150:
        lr *= 0.5e-3
    elif epoch > 100:
        lr *= 1e-3
    elif epoch > 70:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                dropout=False):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if dropout:
            x = Dropout(0.4)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
        if dropout:
            x = Dropout(0.4)(x)
    return x


def resnet_v1(input_shape, depth):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                            dropout=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                            dropout=False)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                                
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Dense(2048,activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.4)(x)
    y = Flatten()(x)

    # Softmax outputs for length, and each of 3 digits on the display
    length = Dense(2, activation='softmax',name='length')(y) # assume possible length is 
    # only two or three (two values)
    digit_1 = Dense(4, activation='softmax',name='digit1')(y) # we have two-digits,
    #100s, 200s, 300s values -> four possibilities
    # in case of the two-digit number a represents "missingness"
    digit_2 = Dense(10, activation='softmax',name='digit2')(y)
    digit_3 = Dense(10, activation='softmax',name='digit3')(y)

    outputs = [length, digit_1, digit_2, digit_3] 
    # outputs will be a list of 
    # probabilities for length and each digit

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    
    return model



# Create CNN model.
model = resnet_v1(input_shape=input_shape, depth=depth)
    
    
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models_attempt6_June29')
model_name = 'Single_digits_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             mode = 'auto',
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=4,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=10,
        # randomly shift images horizontally
        width_shift_range=0.2,
        # randomly shift images vertically
        height_shift_range=0.2,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True)
    
    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    def generate_data_generator(generator, X, Y1, Y2, Y3, Y4, batch_size):
        genXY1 = generator.flow(X, Y1, batch_size=batch_size, seed=7)
        genY2 = generator.flow(X, Y2, batch_size=batch_size, seed=7)
        genY3 = generator.flow(X, Y3, batch_size=batch_size, seed=7)
        genY4 = generator.flow(X, Y4, batch_size=batch_size, seed=7)
        while True:
                Xi, Yi1 = genXY1.next()
                #print(len([Xi, Yi1]))
                _ , Yi2 = genY2.next()
                _ , Yi3 = genY3.next()
                _ , Yi4 = genY4.next()

                yield (Xi, {'length':Yi1, 'digit1':Yi2, 'digit2': Yi3, 'digit3': Yi4})
                
    # since we have a multi-output model, y_train_enc and y_test_enc need to be lists

    # Fit the model on the batches generated by datagen.flow().
    
    
    model.fit_generator(generate_data_generator(datagen,x_train, length_train, digit1_train, digit2_train, digit3_train,\
                        batch_size=batch_size),steps_per_epoch = len(x_train) / batch_size, validation_data=(x_test, [length_test, digit1_test, digit2_test, digit3_test]),
                        epochs=epochs, verbose=1, workers=4, callbacks=callbacks)
#     model.fit(x_train, [length_train, digit1_train, digit2_train, digit3_train],\
#               batch_size=batch_size,validation_data=(x_test, [length_test, digit1_test, digit2_test, digit3_test]), \
#               epochs=epochs, verbose=1,callbacks=callbacks)

save_path = 'saved_architectures_xtrain_means/'
model.save(save_path + 'attempt_6_resnet_architecture_20-400mgDl.h5')

# Save the train mean
np.save('saved_architectures_xtrain_means/'+'x_train_mean_attempt_6.npy', x_train_mean)
