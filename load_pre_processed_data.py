#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:34:22 2021

@author: avinashkumarmishra
"""

from load_data import load_file_data

def build_train_images(train_images_file):

    train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
    
    return train_images
        
        
# load clean descriptions into memory
def load_clean_descriptions_with_pad(filename, dataset):
   
    doc = load_file_data(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line with white space
        tokens = line.split()
        # separate image-id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:

            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens with start and end sequence
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            descriptions[image_id].append(desc)
            
    return descriptions



train_images = build_train_images('train_data/images_name.txt')
train_descriptions = load_clean_descriptions_with_pad('train_data/descriptions.txt', train_images)
print('Captions train :: {0}'.format( len(train_descriptions)))
        
        