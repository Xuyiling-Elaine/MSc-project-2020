#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:38:40 2020

@author: wangxuyiling
"""


from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# load the example image
img = load_img('/Users/wangxuyiling/Desktop/augmentation/aa.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)

# create image data augmentation generator
datagen = ImageDataGenerator(
    fill_mode = 'constant', cval = 0,
    shear_range = 10)

# prepare iterator
it = datagen.flow(samples, batch_size=1)

# generate and plot the results
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)

pyplot.show()