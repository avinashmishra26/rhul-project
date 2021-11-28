#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:14:43 2021

@author: avinashkumarmishra
"""
from load_pre_processed_data import load_clean_descriptions_with_pad, build_train_images

import numpy as np
from time import time

from pickle import dump, load

from keras.models import Model

from keras.preprocessing import image

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

#Inception v3 needs images of 299x299
def pre_process_img_inceptionv3(imagePath):
    img_resize = image.load_img(imagePath, target_size=(299, 299))
    
    output = image.img_to_array(img_resize)

    output = np.expand_dims(output, axis=0)

    output = preprocess_input(output)
    
    return output

# Function to encode a given image into a vector of size (2048, )
def encode(imagePath):
    image = pre_process_img_inceptionv3(imagePath)
    feature_vector = model_with_transfer_learning.predict(image)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1]) # reshape to (2048, )
    return feature_vector


print('in train_incpetionV3_model')


#Without function
model_Iv3 = InceptionV3(weights='imagenet')
# transfer learning by removing softmax layer
model_with_transfer_learning = Model(model_Iv3.input, model_Iv3.layers[-2].output)

##call this to train all the training images
#training_images()

def training_images():
    train_images = build_train_images('train_data/images_name.txt')
     
    start = time()
    encoding_train_img = {}
    for img in train_images:
        encoding_train_img[img] = encode('train_data/images/'+img+'.jpg')
    print('Time taken to encode all images: {0} '.format(time()-start))
    
    
    # Save the encoding images as the training features into the disk
    with open("train_data/encoded_flickr_training_images.pkl", "wb") as encoded_pickle:
        dump(encoding_train_img, encoded_pickle)
