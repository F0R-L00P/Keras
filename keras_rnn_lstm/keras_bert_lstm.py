import os
import pickle
import numpy as np
import pandas as pd
from tkinter import N
from tqdm import tqdm
from typing import List

import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sentence_transformers import SentenceTransformer


import datasets
from datasets import load_dataset

dataset = load_dataset("imdb")

df_train = pd.DataFrame(dataset['train'][0:10])
df_test = pd.DataFrame(dataset['test'][0:10])

df_train
df_test
#######################
#######################
model = SentenceTransformer('bert-base-uncased')

sentence = ['This framework generates embeddings for each input sentence']
embedding = model.encode(sentence)

embedding = model.encode(
    dataset['train']['text'][0:10],
    batch_size=8,
    show_progress_bar=True,
    output_value='token_embeddings',
    device=torch.device('cuda')
)

embedding
cpu_embeddings = [i.cpu() for i in embedding]

# building huggingface dataset
subset_train_data = datasets.Dataset.from_dict({'train_embeddings': embedding})

# concatenate the embeddings dictionary to huggingface dictionary
subset_dataset = datasets.concatenate_datasets(
    [dataset['train'][0:10], subset_train_data], axis=1)

df_train['embeddings'] = cpu_embeddings

# wb = write binary
with open('train_dataset.pickle', 'wb') as handle:
    pickle.dump(new_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

#############################
######### LSTM MODEL##########
#############################
# load pickle file
with open('train_dataset.pickle', 'rb') as pk_file:
    df = pickle.load(pk_file)


model = keras.Sequential(
    [
        keras.Input(shape=(8,)),  # define input in model, skip build
        layers.LSTM(7)
        layers.Dense(50, activation="relu", name="layer1"),
        layers.Dense(1, activation="sigmoid", name="layer2"),
        #        layers.Dense(4, name="layer3"),
    ]
)
