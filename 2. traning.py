#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 19:41:54 2020

@author: wangxuyiling
"""

import keras
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# build up VGG16 model architecture
inputs = Input(shape=(224,224,3))
conv2d_1 = Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu")(inputs)
conv2d_2 = Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu")(conv2d_1)
maxpool2d_1 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_2)

conv2d_3 = Conv2D(filters=128,kernel_size=(3,3), padding="same", activation="relu")(maxpool2d_1)
conv2d_4 = Conv2D(filters=128,kernel_size=(3,3), padding="same", activation="relu")(conv2d_3)
maxpool2d_2 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_4)

conv2d_5 = Conv2D(filters=256,kernel_size=(3,3), padding="same", activation="relu")(maxpool2d_2)
conv2d_6 = Conv2D(filters=256,kernel_size=(3,3), padding="same", activation="relu")(conv2d_5)
conv2d_7 = Conv2D(filters=256,kernel_size=(3,3), padding="same", activation="relu")(conv2d_6)
maxpool2d_3 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_7)

conv2d_8 = Conv2D(filters=512,kernel_size=(3,3), padding="same", activation="relu")(maxpool2d_3)
conv2d_9 = Conv2D(filters=512,kernel_size=(3,3), padding="same", activation="relu")(conv2d_8)
conv2d_10 = Conv2D(filters=512,kernel_size=(3,3), padding="same", activation="relu")(conv2d_9)
maxpool2d_4 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_10)

conv2d_11 = Conv2D(filters=512,kernel_size=(3,3), padding="same", activation="relu")(maxpool2d_4)
conv2d_12 = Conv2D(filters=512,kernel_size=(3,3), padding="same", activation="relu")(conv2d_11)
conv2d_13 = Conv2D(filters=512,kernel_size=(3,3), padding="same", activation="relu")(conv2d_12)
maxpool2d_5 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_13)

flatten1 = Flatten()(maxpool2d_5)

dense1 = Dense(units=4096,activation="relu")(flatten1)
dense2 = Dense(units=4096,activation="relu")(dense1)
dense3 = Dense(units=1000,activation="softmax")(dense2)

vgg16 = Model(inputs = inputs, outputs = dense3)

#load VGG16 weights pre-trained on ImageNet
vgg16.load_weights('D:/ElaineWang/weights/vgg16_weights.h5')

# layer trainable and untrainable settings
for layer in vgg16.layers[:]:
    layer.trainable = True
#for layer in vgg16.layers[:3]:
#    layer.trainable = False

# create a new changeable classifer for VGG with two nodes as the putput
dense_1 = Dense(units=4096,activation="relu")(flatten1)
dense_2 = Dense(units=4096,activation="relu")(dense_1)
dense_3 = Dense(units=2,activation="softmax")(dense_2)

# model used in the experiment
model2 = Model(inputs = vgg16.input, outputs = dense_3)

# compile the model
from keras.optimizers import Adam
model2.compile(Adam(lr = 0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# set the training and validation data
num_train = (350+1611)*9 + (350+1610)   # now is 10 portions, which is all the training set
num_valid = 511+2350
#num_test = 511+2350
ba_size = 64

# doing rescaling and data augmentation for training set only
train_gen = ImageDataGenerator(
    rotation_range = 180,
    zoom_range = [0.7,1], 
    brightness_range = [0.8, 1.2],
    width_shift_range = [-0.1, 0.1],
    height_shift_range = [-0.1, 0.1],
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'constant', cval = 0,
    rescale = 1./255)

# doing rescaling for validation set
rescale_gen = ImageDataGenerator(rescale = 1./255)

# generate training and validation batches
train_batches = train_gen.flow_from_directory(
    'D:/ElaineWang/full_data/train/10', 
    target_size=(224,224), 
    batch_size = ba_size)
valid_batches = rescale_gen.flow_from_directory(
    'D:/ElaineWang/full_data/valid', 
    target_size=(224,224), 
    batch_size = ba_size)

# train the model with early_stop and checkpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import math
early_stop = EarlyStopping(monitor='val_loss',patience=12, restore_best_weights = True)
checkpoint = ModelCheckpoint(
    "D:/ElaineWang/model/a1.h5", 
    monitor='val_loss', 
    save_best_only=True, save_weights_only=False, mode='auto')
csv_logger = CSVLogger('D:/ElaineWang/a1.csv', append=True, separator=';')
hist = model2.fit_generator(
    train_batches, 
    steps_per_epoch = math.ceil(num_train/ba_size), 
    validation_data = valid_batches, 
    validation_steps = math.ceil(num_valid/ba_size), 
    epochs = 150, 
    callbacks=[checkpoint, early_stop, csv_logger], 
    verbose = 2)

    
    
    
    
    
    
    
    
    
    