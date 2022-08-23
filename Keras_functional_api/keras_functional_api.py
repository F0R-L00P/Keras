from gc import callbacks
from tabnanny import verbose
import pandas as pd
import pydot
import graphviz

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        keras.Input(shape=(8,)), # define input in model, skip build
        layers.Dense(50, activation="relu", name="layer1"),
        layers.Dense(1, activation="sigmoid", name="layer2"),
#        layers.Dense(4, name="layer3"),
    ]
)

#check model weights for shape
model.weights
#check layers
#layer params = weights+bias
model.layers[0].weights
model.layers[0].bias

#obtain summary for input-output
model.summary()

# Call model on a test input
x = tf.ones((3, 3))
prediction_y = model(x)


##########################################
# FUNCTIONAL API
#########################################
# define input layer on 1 sample
# defining a node
inputs = keras.Input(shape=(8,))
# define the next layer in the network but not yet placed!
dense1 = layers.Dense(64, activation="relu", name='dense1')
dense2 = layers.Dense(32, activation="relu", name='dense2')
# connecting the layers (two branchs)
inter_layer1 = dense1(inputs)
inter_layer2 = dense2(inputs)

last_layer1 = layers.Dense(2, name='layer1')(inter_layer1)
last_layer2 = layers.Dense(2,name='layer2')(inter_layer2)

output = keras.layers.Concatenate(axis=0)([last_layer1, last_layer2])

model = keras.Model(inputs=inputs, outputs=output, name="my_res_model")

model.summary()

X = tf.ones((50, 784))
model(X).shape

##########################################
# FUNCTIONAL API
# Random Net on Titanic Data
#########################################
df = pd.read_csv(r'titanic\train.csv')
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df = pd.get_dummies(df, prefix=['Sex', 'Embark'], drop_first=True)
df['Age'].fillna((df['Age'].mean()), inplace=True)
df.isna().sum()
df.head()

X = df.loc[:, df.columns != 'Survived']
y = df['Survived']

###########################
#KERAS NN MODEL
###########################
# define input layer
titanic_input = keras.Input(shape=(8,))

#64 neuron connected layers
dense1 = layers.Dense(64, activation="relu", name='64dense1')
dense2 = layers.Dense(64, activation="relu", name='64dense2')
dense3 = layers.Dense(100, activation="relu", name='64dense3')
dense4 = layers.Dense(50, activation="relu", name='64dense4')
dense5 = layers.Dense(2, activation="relu", name='64dense5')

#32 neuron connected layers
dense11 = layers.Dense(32, activation="relu", name='32dense1')
dense21 = layers.Dense(32, activation="relu", name='32dense2')
dense31 = layers.Dense(100, activation="relu", name='32dense3')
dense41 = layers.Dense(50, activation="relu", name='32dense4')
dense51 = layers.Dense(2, activation="relu", name='32dense5')

# parallel passing of data
# 64 layer
internal64_dense1 = dense1(titanic_input)
internal64_dense2 = dense2(internal64_dense1)
internal64_dense3 = dense3(internal64_dense2)
internal64_dense4 = dense4(internal64_dense3)
internal64_dense5 = dense5(internal64_dense4) #->to output
# 32 layer
internal32_dense1 = dense11(titanic_input)
internal32_dense2 = dense21(internal32_dense1)
internal32_dense3 = dense31(internal32_dense2)
internal32_dense4 = dense41(internal32_dense3)
internal32_dense5 = dense51(internal32_dense4)#-> to output

#64-32 combination
dense_mix1 = layers.Dense(2, activation="relu", name='inner_concat')

#interlayer concatination
inter_layer = keras.layers.Concatenate(axis=1)(
    [
        internal64_dense2, 
        internal32_dense2
    ]
)
inter_layer_mix1 = dense_mix1(inter_layer)

# dense layer concatination
# output of 64n layer and 32n layer
dense_64_32_concat_layer = keras.layers.Concatenate(axis=1)(
    [
        internal64_dense5, 
        internal32_dense5
    ]
)
dense_mix2 = layers.Dense(2, activation="relu", name='outer_concat')
inter_layer_mix2 = dense_mix2(dense_64_32_concat_layer)

# final concatination
titanic_output = keras.layers.Concatenate(axis=1)(
    [
        inter_layer_mix1, 
        inter_layer_mix2
    ]
)

dense_mix3 = layers.Dense(1, activation="sigmoid", name='outer_inner_concat')
final_output_mix3 = dense_mix3(titanic_output)

# tetsing model
model = keras.Model(
    inputs=titanic_input, 
    outputs=final_output_mix3, 
    name="titanic_res_model"
)

test = tf.ones((1000, 8))
model(test).shape

keras.utils.plot_model(model, "titanic.png", show_shapes=True)
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x=X,
    y=y,
    batch_size=5,
    epochs=100,
    verbose=2,
    validation_split=.15,
    callbacks=[early_stop],
    shuffle=True,
)

model.summary()
