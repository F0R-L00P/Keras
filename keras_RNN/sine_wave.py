import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.core import Activation
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,\
    confusion_matrix

random.seed(7)
np.random.seed(7)
tensorflow.random.set_seed(7)
####################################################
# lets generate X values and corresponding sin at y
X = np.linspace(
                    start=0, 
                    stop=100, 
                    num=700
                )
y = np.sin(X)

# visualize values
plt.plot(X, y);

# process as dataframe
df = pd.DataFrame(data=y, index=X, columns=['sine_values'])
df.head()
round(df)
# break dataframe to seperate 
# check number of instances
len(df)

# setting-up test size
test_percent = 0.20
# get data %
test_index = int(len(df) * test_percent)
# get tain index
train_index = int(len(df)) - test_index

train_data = df.iloc[:train_index]
test_data = df.iloc[train_index:]

# check train test split visual
plt.plot(train_data)
plt.plot(test_data);

# scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

##################################################
length = 50
batch = 1

generator = TimeseriesGenerator(
                                data=train_data,
                                targets=train_data,
                                length=length,
                                batch_size=batch
                                )

#testing len and batch
len(train_data)
len(generator) #lower len as the batch is of len 2

X, y = generator[0]

# given the n points provided for X
#   as per the length specifid in the parameter
#   predict the third point y
print(X, y)

###################################################
# modeling

