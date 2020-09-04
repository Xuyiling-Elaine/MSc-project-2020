#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:34:58 2020

@author: wangxuyiling
"""


# load the pre-trained model transported from remote computer
from tensorflow.keras.models import load_model
model1 = load_model('/Users/wangxuyiling/Desktop/saved_model/a1.h5')

# rescaling for test set
from keras.preprocessing.image import ImageDataGenerator
rescale_gen = ImageDataGenerator(rescale = 1./255)

# set the test batches
test_batches = rescale_gen.flow_from_directory(
    '/Users/wangxuyiling/Desktop/test',
    target_size=(224,224), 
    class_mode=None, shuffle = False,
    batch_size=1)

# make predictions
predict = model1.predict_generator(test_batches, verbose = 0)
prediction = predict[:,1]
real_label = test_batches.classes

# rounded predictions
import numpy as np
round_predictions = np.round(prediction)

# calculate and plot the ROC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
fpr, tpr, thre_roc = roc_curve(real_label, prediction)
pyplot.plot(fpr, tpr, marker='.', label='ROC')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# calculate and print Area Under full/part of ROC
auc_roc = roc_auc_score(real_label, prediction)
auc_roc100 = roc_auc_score(real_label, prediction, max_fpr=1)
auc_roc95 = roc_auc_score(real_label, prediction, max_fpr=0.05)
auc_roc90 = roc_auc_score(real_label, prediction, max_fpr=0.1)
auc_roc80 = roc_auc_score(real_label, prediction, max_fpr=0.2)
print('AU ROC:', auc_roc)
print('95:', auc_roc95)
print('90:', auc_roc90)
print('80:', auc_roc80)

# calculate and plot the PRC
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
precision, recall, thre_prc = precision_recall_curve(real_label, prediction)
pyplot.plot(recall, precision, marker='.', label='PRC')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# calculate and print Area Under PRC
auc_prc = auc(recall, precision)
print('AU PRC:', auc_prc)

# compute and print f1 score
f1= f1_score(real_label, round_predictions)
print('f1=', f1)

# calculate and print: tp, tn, fp, fn for confusion matrix.
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(2861):
    if (round_predictions[i] == 0 and real_label[i] == 0):
            tp = tp + 1
    if (round_predictions[i] == 1 and real_label[i] == 1):
            tn = tn + 1
    if (round_predictions[i] == 1 and real_label[i] == 0):
            fn = fn + 1
    if (round_predictions[i] == 0 and real_label[i] == 1):
            fp = fp + 1
print('tp:', tp)
print('tn:', tn)
print('fp:', fp)
print('fn:', fn)

