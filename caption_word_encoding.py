#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:07:15 2021

@author: avinashkumarmishra
"""
from load_pre_processed_data import load_clean_descriptions_with_pad, build_train_images
import os
import numpy as np
from numpy import array

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class WordEmbedding(object):
 
    # we have generated word with threasold limit 10, 20 as well. 
    word_cnt_threshold = 5 
    embedding_dimension = 200
    
    def __init__(self):
        self.idx_to_word = {}
        self.word_to_idx = {}
        self.word_embeddings = {}
        self.max_length = 0
        self.total_vocab_size = 0
        
        
    def max_length_sentence(self, captions):
        all_captions = list()
        for key in captions.keys():
            [all_captions.append(d) for d in captions[key]]
         
        self.max_length = max(len(d.split()) for d in all_captions)
        return self.max_length
        
    def evaluate_word_map(self, train_descriptions_captions):
        # read and collect a list of all the training captions
        captions_list = []
        for img_key, captions in train_descriptions_captions.items():
            for caption in captions:
                captions_list.append(caption)
        
        #print(len(all_train_captions))    #-> total Caption (40455)
        
        token_counts = {}
        for each_caption in captions_list:
            for token in each_caption.split():
                token_counts[token] = token_counts.get(token, 0) + 1
        
        threshold_vocab = [w for w in token_counts if token_counts[w] >= WordEmbedding.word_cnt_threshold]
        print('Total words {0} ::: and threashold words {1}'.format( len(token_counts), len(threshold_vocab)))
        
        index = 1
        for vocab in threshold_vocab:
            self.word_to_idx[vocab] = index
            self.idx_to_word[index] = vocab
            index += 1
        
        # append for 0's
        self.total_vocab_size = len(self.word_to_idx) + 1 
       
        print('Considered vocabulary Size : {0}'.format(len(self.idx_to_word)))
        
    def glove_representation(self):
        glove_dir = 'glove/'
        self.word_embeddings = {} # empty dictionary
        
        with open(os.path.join(glove_dir, 'glove.6B.200d.txt'), 
                  encoding="utf-8") as file:     
            for line in file:
                values = line.split()
                word = values[0]
                embedding_vector = np.asarray(values[1:], dtype='float32')
                self.word_embeddings[word] = embedding_vector
            #print('Total Words vectors :: {0}'.format(len(self.word_embeddings)))
            
    def get_embedding_matrix(self):      
        embedding_matrix = np.zeros((self.total_vocab_size, WordEmbedding.embedding_dimension))
        for token, index in self.word_to_idx.items():
            word_vector = self.word_embeddings.get(token)
            if word_vector is not None:
                embedding_matrix[index] = word_vector
        return embedding_matrix
            
    
    #img_caption_mapping : image sentences, image_encoding : 
    def word_embedding_generator(self, img_caption_mapping, image_encoding, word_to_idx, max_length, num_img_per_batch):
        image_X1, word_embed_X2, y_target = list(), list(), list()
        iter_epoch = 0
        # starting the generator logic on datapoints
        while 1:
            for img_name, caption_list in img_caption_mapping.items():
                iter_epoch += 1
                # retrieve the photo encoded feature derived from Inception v3
                image_encode = image_encoding[img_name]
                for caption in caption_list:
                    tokens = [word_to_idx[word] for word in caption.split(' ') if word in word_to_idx]
                    # split captions into multiple token of x, y pairs
                    for i in range(1, len(tokens)):
                        # split into previous sequence as input and next token as output
                        input_seq, out_seq = tokens[:i], tokens[i]
                        # pading input sequence to max-length of caption
                        input_seq = pad_sequences([input_seq], maxlen = max_length)[0]
                        # encoding target or output sequence
                        out_seq = to_categorical([out_seq], num_classes = self.total_vocab_size)[0]
                        # store
                        image_X1.append(image_encode)
                        word_embed_X2.append(input_seq)
                        y_target.append(out_seq)
                # yield image batch data
                if iter_epoch == num_img_per_batch:
                    iter_epoch = 0
                    yield ([array(image_X1), array(word_embed_X2)], array(y_target))
                    #re-initialize the yield variables
                    image_X1, word_embed_X2, y_target = list(), list(), list()
            
        

train_images = build_train_images('train_data/images_name.txt')
train_descriptions = load_clean_descriptions_with_pad('train_data/descriptions.txt', train_images)

e = WordEmbedding()
e.max_length_sentence(train_descriptions)
e.glove_representation()
e.evaluate_word_map(train_descriptions)

embedding_matrix = e.get_embedding_matrix()

