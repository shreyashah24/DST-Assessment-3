# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:40:05 2023

@author: Danie
"""

import tensorflow 
import os
from skimage import io 
from PIL import Image
import random 
import numpy as np
import matplotlib.pyplot as plt  

# Importing and Loading the data into a data frame
dataset_path_train = 'Data/Training/'
dataset_path_test = 'Data/Testing/'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

## Importing training
training = {}
for i in range(len(class_names)):
    tumor_path_train = os.path.join(dataset_path_train, class_names[i])
    training[class_names[i]+'_train'] = [Image.open(os.path.join(tumor_path_train, image)) for image in os.listdir(tumor_path_train)]
    
## Importing test
test = {}
for i in range(len(class_names)):
    tumor_path_test = os.path.join(dataset_path_test, class_names[i])
    test[class_names[i]+'_test'] = [Image.open(os.path.join(tumor_path_test, image)) for image in os.listdir(tumor_path_test)]
    
    # Training to array
for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        training[class_names[i]+'_train'][j] = np.array(training[class_names[i]+'_train'][j])
        
## Test to array
for i in range(len(class_names)):
    for j in range(len(test[class_names[i]+'_test'])):
        test[class_names[i]+'_test'][j] = np.array(test[class_names[i]+'_test'][j])
        
## training set normalising
for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        training[class_names[i]+'_train'][j] = (training[class_names[i]+'_train'][j] - np.min(training[class_names[i]+'_train'][j])) / (np.max(training[class_names[i]+'_train'][j]) - np.min(training[class_names[i]+'_train'][j]))
        
## test set normalising
for i in range(len(class_names)):
    for j in range(len(test[class_names[i]+'_test'])):
        test[class_names[i]+'_test'][j] = (test[class_names[i]+'_test'][j] - np.min(test[class_names[i]+'_test'][j])) / (np.max(test[class_names[i]+'_test'][j]) - np.min(test[class_names[i]+'_test'][j]))
        
new_training = training

for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        new_training[class_names[i]+'_train'].append(np.fliplr(training[class_names[i]+'_train'][j]))
        new_training[class_names[i]+'_train'].append(np.flipud(training[class_names[i]+'_train'][j]))
        
for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        gaussian = np.random.normal(0, 1, training[class_names[i]+'_train'][j].shape)
        n2 = np.clip((training[class_names[i]+'_train'][j] + gaussian*0.2),0,1)
        new_training[class_names[i]+'_train'].append(n2)
        
for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        new_training[class_names[i]+'_train'].append(training[class_names[i]+'_train'][j].rotate(90))
        new_training[class_names[i]+'_train'].append(training[class_names[i]+'_train'][j].rotate(180))
        new_training[class_names[i]+'_train'].append(training[class_names[i]+'_train'][j].rotate(270))
        
        
## Array to training
for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        new_training[class_names[i]+'_train'][j] = Image.fromarray(training[class_names[i]+'_train'][j])
        
## Array to test
for i in range(len(class_names)):
    for j in range(len(test[class_names[i]+'_test'])):
        test[class_names[i]+'_test'][j] = Image.fromarray(test[class_names[i]+'_test'][j])
        
def random_crop(image):
    cropped_image = tensorflow.image.random_crop(image, size=[300, 300, 3])
    resized = tensorflow.image.resize(cropped_image, (512, 512))
    resized_image_pil = tensorflow.keras.utils.array_to_img(resized)
    return resized_image_pil

for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        for k in range(2):
            new_training[class_names[i]+'_train'].append(random_crop(training[class_names[i]+'_train'][j]))
            
#for i in range(len(class_names)):
#    for j in range(len(training[class_names[i]+'_train'])):
#        new_training.append(training[class_names[i]+'_train'][j].filter(ImageFilter.BoxBlur(4)))
        
## Training data
train_path = 'NewData/Training/'
for i in range(len(class_names)):
    for j in range(len(training[class_names[i]+'_train'])):
        new_training[class_names[i]+'_train'][j].save(train_path+class_names[i]+'/'+str(j)+'.jpg')
        
## Test data
test_path = 'NewData/Testing/'
for i in range(len(class_names)):
    for j in range(len(test[class_names[i]+'_train'])):
        test[class_names[i]+'_test'][j].save(test_path+class_names[i]+'/'+str(j)+'.jpg')

