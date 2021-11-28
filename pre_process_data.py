#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:54:06 2021

@author: avinashkumarmishra
"""

from load_data import load_descriptions

import string
#Visulization of data and other metrics
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Data Preprocessing 
def pre_process_caption(captions):
    punctua_table = str.maketrans('', '', string.punctuation)
    for key, desc_data_list in captions.items():
        for i in range(len(desc_data_list)):
            desc = desc_data_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(punctua_table) for w in desc]
            desc = [word for word in desc if len(word)>1 or word == 'a']
            desc = [word for word in desc if word.isalpha()]
            desc_data_list[i] =  ' '.join(desc)
            
    return captions


def total_vocabulary(captions):
    all_vocabs = set()
    for key in captions.keys():
        [all_vocabs.update(c.split()) for c in captions[key]]
    return all_vocabs


def use_word_cloud(all_words):

    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white')\
                         .generate(' '.join(i for i in all_words))
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Vocabulary Frequency', fontsize=20)
    plt.axis("off")
    plt.show()
    
# Get top 30 words by frequency.
def vocabs_cnt(descriptions):
    totalCaption = 0;
    all_vocabs_dict = dict()
    for key, caption in descriptions.items():
        for tokens in caption:
            totalCaption +=1
            for token in tokens.split():
                if(all_vocabs_dict.get(token, -1) != -1):
                    all_vocabs_dict[token] = all_vocabs_dict[token] + 1;
                else:
                    all_vocabs_dict[token] = 1;
            
    vals = sorted(all_vocabs_dict.items(),key=lambda x:x[1],reverse=True)
    words = [x[0] for x in vals[:32]]
    cnts = [x[1] for x in vals[:32]]
    plt.figure(figsize=(20,5))
    plt.bar(words,cnts)
    print('Total Caption {0}'.format(totalCaption))

def execute_pre_process():
    filePath = "train_data/captions.txt"
    descriptions = load_descriptions(filePath)
    pre_process_caption(descriptions)
    return descriptions
    
#print method
def pre_process_display():
    #printing data to check the response
    filePath = "train_data/captions.txt"
    descriptions = load_descriptions(filePath)
    pre_process_caption(descriptions)
    print(descriptions['1000268201_693b08cb0e'])
    print(descriptions['1001773457_577c3a7d70'])
    vocabulary = total_vocabulary(descriptions)
    print('Vocabulary Size: {0}'.format(len(vocabulary)))
    
    use_word_cloud(vocabulary)
    vocabs_cnt(descriptions)
    