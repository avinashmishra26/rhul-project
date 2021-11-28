#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:27:16 2021

@author: avinashkumarmishra
"""

# load caption file into memory and return to the caller
def load_file_data(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text

#load and map the captions of an image to dict
def load_descriptions(filePath):
    doc_data = load_file_data(filePath)
    #print(doc_data[:500])
    map_data = dict()
    for idx, data in enumerate(doc_data.split('\n')):
        if (idx == 0) or len(data) < 3:
            continue
        
        img_tokens = data.split(',')

        image_id, image_text = img_tokens[0], img_tokens[1:]
        
        image_id = image_id.split('.')[0]

        image_text = ' '.join(image_text)
        
        if image_id not in map_data:
            map_data[image_id] = list()

        map_data[image_id].append(image_text)
        
    return map_data


#print method
def load_file_display():
    #printing data to check the response
    filePath = "train_data/captions.txt"
    descriptions = load_descriptions(filePath)
    print(len(descriptions))
    print(list(descriptions.keys())[:4])
    print()
    print(descriptions['1000268201_693b08cb0e'])
    print()
    print(descriptions['1001773457_577c3a7d70'])