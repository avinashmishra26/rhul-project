#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:54:06 2021

@author: avinashkumarmishra
"""

from pre_process_data import execute_pre_process

# save descriptions to file, one per line
def save_captions(captions, filename):
    lines = list()
    for key, desc_list in captions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# save image name to file, one per line
def save_img_cap(img_Keys, filename):
    lines = list()
    for desc_list in img_Keys:
        lines.append(desc_list)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def save_training_helper() :
    captions = execute_pre_process()
    save_captions(captions, 'train_data/descriptions.txt')
    save_img_cap(set(captions.keys()), 'train_data/images_name.txt')
    print('Dataset: {0}'.format(len(captions.keys())))
    

#prepare and save images and its caption
save_training_helper()