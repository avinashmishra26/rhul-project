#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 10:07:15 2021

@author: avinashkumarmishra
"""
from caption_word_encoding import WordEmbedding
from load_pre_processed_data import load_clean_descriptions_with_pad, build_train_images

from pickle import dump, load

from keras import Input, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization

from keras.layers.merge import add
from keras.models import Model

class Image_Caption():
    
    def __init__(self, word_embedding_matrix, word_embedding, embedding_dimension = 200):
        self.word_embedding = word_embedding
        
        img_inputs = Input(shape=(2048,))
        img_inputs_dp = Dropout(0.5)(img_inputs)
        img_inputs_dense = Dense(256, activation='relu')(img_inputs_dp)
        token_inputs = Input(shape=(self.word_embedding.max_length,))
        token_inputs_emd = Embedding(word_embedding.total_vocab_size, embedding_dimension, mask_zero=True)(token_inputs)
        token_inputs_dp = Dropout(0.5)(token_inputs_emd)
        token_inputs_lstm = LSTM(256)(token_inputs_dp)
        merge_img_token = add([img_inputs_dense, token_inputs_lstm])
        merge_img_dense = Dense(256, activation='relu')(merge_img_token)
        model_output = Dense(word_embedding.total_vocab_size, activation='softmax')(merge_img_dense)
        self.model = Model(inputs=[img_inputs, token_inputs], outputs = model_output)
        self.model.layers[2].set_weights([word_embedding_matrix])
        self.model.layers[2].trainable = False
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.optimizer.lr = 0.0001
        
        
    def model_summary(self):
        print(self.model.summary())
        
    def exec_model(self, train_descriptions, train_features, steps, number_img_per_batch = 3, epochs = 10):
        for i in range(epochs):
            generator = self.word_embedding.word_embedding_generator(train_descriptions, train_features, 
                                                                     self.word_embedding.word_to_idx, 
                                                                     self.word_embedding.max_length, 
                                                                     number_img_per_batch)
            self.model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
            self.model.save('./model_weights/model_' + str(i) + '.h5')
        
        
        
def train_the_model():       
    train_images = build_train_images('train_data/images_name.txt')
    train_descriptions = load_clean_descriptions_with_pad('train_data/descriptions.txt', train_images)
    
    e = WordEmbedding()
    max_length_sentence = e.max_length_sentence(train_descriptions)
    e.glove_representation()
    e.evaluate_word_map(train_descriptions)
    
    word_embedding_matrix = e.get_embedding_matrix()
    
    train_features = load(open("train_data/encoded_flickr_training_images.pkl", "rb"))
    print('Photos: train=%d' % len(train_features))
    #print('Photo: train= {0}'.format(train_features[0]))
    
    number_img_per_batch = 3
    steps = len(train_descriptions)//number_img_per_batch
    
    i_c = Image_Caption(word_embedding_matrix, e, 200)
    i_c.exec_model(train_descriptions, train_features, steps, number_img_per_batch)
    
    #training again
    i_c.exec_model(train_descriptions, train_features, steps)
    
    #changing learning rate
    i_c.model.optimizer.lr = 0.0001
    number_img_per_batch = 6
    steps = len(train_descriptions)//number_img_per_batch
    i_c.exec_model(train_descriptions, train_features, steps, number_img_per_batch)

