# STAPEL LIBRARIES
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DEEP LEARNING LIBRARIES
import keras
import tensorflow
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Activation, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# MACHINE LEARNING LIBRARIES
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,\
    confusion_matrix

# SETTING ALL SEED LEVELS
random.seed(7)
np.random.seed(7)
tensorflow.random.set_seed(7)
####################################################
# lets generate X values and corresponding sin at y
df = pd.read_csv(
                r'GitHub\Keras\keras_RNN\timeseries_retail_data.csv',
                parse_dates=True,
                index_col='DATE'
                )

df.head()
# visualize values
df.plot(figsize=(12, 8));

# check number of instances
len(df)

# setting-up test size
test_percent = 0.05
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
length = 12 # which is less than the test data length
batch = 1

train_generator = TimeseriesGenerator(
                                data=train_data,
                                targets=train_data,
                                length=length,
                                batch_size=batch
                                )

#testing len and batch
len(train_data)
len(train_generator) #lower len as the batch is of len 2

X, y = train_generator[0]

# given the n points provided for X
#   as per the length specifid in the parameter
#   predict the third point y
print(X, y)

#setting up validation generator
val_generator = TimeseriesGenerator(
                                    data=test_data,
                                    targets=test_data,
                                    length=length,
                                    batch_size=batch
                                    )

X, y = val_generator[0]
print(X, y)
###################################################
n_features=1
stopper = EarlyStopping(
                        monitor='val_loss',
                        patience=2
                        )
# modeling
model = keras.Sequential(
                    [
                        keras.layers.LSTM(
                                            10, 
                                            activation='relu',
                                            #inpout is length of batch 
                                            #by number of features
                                            input_shape=(length,n_features)
                                        ),
                        keras.layers.Dense(1)
                    ]
)

loss_function = keras.losses.MeanSquaredError(
                        name="mse"
)

model.compile(
                optimizer='adam', 
                loss=loss_function, 
                metrics=['accuracy']
            )

# fit model from the built generator
model.fit_generator(
                     train_generator,
                     epochs=20,
                     validation_data=val_generator,
                     callbacks=[stopper],
                     verbose=2
                    )

model.summary()

# get model loss
losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()

# test new instances and make prediction
test_prediction = []
