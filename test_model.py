#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 05 03:16:45 2021

@author: avinashkumarmishra
"""
from train_inceptionV3_model import encode
from caption_word_encoding import WordEmbedding
from time import time
from keras.preprocessing.sequence import pad_sequences
from model_build import Image_Caption
from load_pre_processed_data import load_clean_descriptions_with_pad, build_train_images
from pickle import load
import numpy as np
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

def test_model():
    test_images_file = 'test_img/test_dog.txt'
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
    test_img = []
    
    
    for i in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images
        
    
    start = time()
    encoding_test_img = {}
    for img in test_img:
        encoding_test_img[img] = encode('test_img/'+img)
    print("Time taken in seconds =", time()-start)
    return encoding_test_img
    


#model1 = model
    
def predict_img_txt(image, model, max_length_sentence,  word_to_idx, idx_to_word):
    in_text = 'startseq'
    for i in range(max_length_sentence):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_length_sentence)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

#call this to predict the images
def prepare_test_model():
    train_images = build_train_images('train_data/images_name.txt')
    train_descriptions = load_clean_descriptions_with_pad('train_data/descriptions.txt', train_images)
    
    word_embedding = WordEmbedding()
    max_length_sentence = word_embedding.max_length_sentence(train_descriptions)
    word_embedding.glove_representation()
    word_embedding.evaluate_word_map(train_descriptions)
    
    word_embedding.word_to_idx
    word_embedding.idx_to_word
    

    
    encoding_test_img = test_model()
    z=0
    pic = list(encoding_test_img.keys())[z]
    image = encoding_test_img[pic].reshape((1,2048))
    x=plt.imread('test_img/'+pic)
    plt.imshow(x)
    plt.show()
    print("Caption predict :" ,predict_img_txt(image,model,max_length_sentence,\
                                               word_embedding.word_to_idx,\
                                               word_embedding.idx_to_word))


def build_the_model():
    train_images = build_train_images('train_data/images_name.txt')
    train_descriptions = load_clean_descriptions_with_pad('train_data/descriptions.txt', train_images)
    
    e = WordEmbedding()
    max_length_sentence = e.max_length_sentence(train_descriptions)
    e.glove_representation()
    e.evaluate_word_map(train_descriptions)
    
    embedding_matrix = e.get_embedding_matrix()
    
    train_features = load(open("train_data/encoded_flickr_training_images.pkl", "rb"))
    print('Photos: train=%d' % len(train_features))
    #print('Photo: train= {0}'.format(train_features[0]))
    
    number_img_per_batch = 3
    steps = len(train_descriptions)//number_img_per_batch
    
    i_c = Image_Caption(embedding_matrix, e, 200)
    i_c.exec_model(train_descriptions, train_features, steps, number_img_per_batch)
    
    #training again
    i_c.exec_model(train_descriptions, train_features, steps)
    
    #changing learning rate
    i_c.model.optimizer.lr = 0.0001
    number_img_per_batch = 6
    steps = len(train_descriptions)//number_img_per_batch
    i_c.exec_model(train_descriptions, train_features, steps, number_img_per_batch)
    
    #saving the trained model
    i_c.model.save_weights('./saved_model/img_caption_model.h5')
    
    return i_c.model


def get_saved_model():
    train_images = build_train_images('train_data/images_name.txt')
    train_descriptions = load_clean_descriptions_with_pad('train_data/descriptions.txt', train_images)
    
    e = WordEmbedding()
    max_length_sentence = e.max_length_sentence(train_descriptions)
    e.glove_representation()
    e.evaluate_word_map(train_descriptions)
    
    embedding_matrix = e.get_embedding_matrix()
    
    train_features = load(open("train_data/encoded_flickr_training_images.pkl", "rb"))
    print('Photos: train= {0}'.format(len(train_features)))
    #print('Photo: train= {0}'.format(train_features[0]))
    
    number_img_per_batch = 3
    steps = len(train_descriptions) // number_img_per_batch
    
    i_c = Image_Caption(embedding_matrix, e, 200)
    #loading the saved trained model
    i_c.model.load_weights('./saved_model/img_caption_model.h5')
    
    return i_c.model


#To re-train the model either by building again or loading the existing/saved model
#if
#model = build_the_model() 
#else
model = get_saved_model()


def find_blue_score_sentence(candidate, reference):
    print('Sentence 1-gram: {0}'.format(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))))
    print('Sentence 2-gram: {0}'.format(sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))))
    print('Sentence 3-gram: {0}'.format(sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))))
    print('Sentence 4-gram: {0}'.format(sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))))

    
 