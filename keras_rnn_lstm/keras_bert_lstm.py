import os
import pickle
from tkinter import N
from tqdm import tqdm

import datasets
from datasets import load_dataset

dataset = load_dataset("imdb")

dataset['train']

#######################
# import Flair library
#######################
import torch
from transformers import BertModel, BertTokenizer

# define tokenizer model

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# testing tokenizer
text = "Hello World!, I'm a computer!"
print(
    f'original text: {text}\n',
    f'tokenized text: {tokenizer.tokenize(text)}\n',
    f'token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))}'
)


def transformer_tokenizer(target:str):
    # For every sentence...    
    for sent in target:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        encoded_input = tokenizer(
                                    sent, 
                                    add_special_tokens=True,
                                    padding=True,
                                    return_tensors="pt"
                                )
        output = model(**encoded_input)
    
    return output
        

testing_sent = transformer_tokenizer(dataset['train']['text'][0])



# wb = write binary
with open('train_dataset.pickle', 'wb') as handle:
    pickle.dump(new_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

#############################
#########LSTM MODEL##########
#############################
# load pickle file
with open('train_dataset.pickle', 'rb') as pk_file:
    df = pickle.load(pk_file)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential(
    [
        keras.Input(shape=(8,)), # define input in model, skip build
        layers.LSTM(7)
        layers.Dense(50, activation="relu", name="layer1"),
        layers.Dense(1, activation="sigmoid", name="layer2"),
#        layers.Dense(4, name="layer3"),
    ]
)

