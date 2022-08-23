import os
import pickle
from tqdm import tqdm

import datasets
from datasets import load_dataset

dataset = load_dataset("imdb")

dataset['train']


# import Flair library
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import flair, torch

# since this is small dataset, can process this with cpu
flair.device = torch.device('cpu')

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')

# building function that for given column, it will iterate over each row and embed sentences
def word_embedding(target_feature):
    sentence = Sentence(target_feature)
    # embed the sentence with our document embedding
    glove_embedding.embed(sentence)
    return [token.embedding for token in sentence]

# test embedding
sentence = Sentence(dataset['train']['text'][0])
word_embedding(dataset['train']['text'][0])


embed_list = []
for i in tqdm(dataset['train']['text']):
    embed_list.append(word_embedding(i))

# building huggingface dataset
target_train_data = datasets.Dataset.from_dict({'train_embeddings':embed_list})

# concatenate the embeddings dictionary to huggingface dictionary
new_dataset = datasets.concatenate_datasets([dataset['train'], target_train_data], axis=1)

new_dataset['train_embeddings'][0]

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

