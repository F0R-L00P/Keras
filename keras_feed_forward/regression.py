import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from sklearn.metrics import r2_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping


housing = fetch_openml(name="house_prices", as_frame=True)

# obtain feature and target
for key, value in housing.items():
    if key == 'data':
        X = pd.DataFrame(value)
    elif key == 'target':
        y = value

X = X[['MSSubClass', 'LotArea', 'OverallQual', 
        'YearBuilt', 'YearRemodAdd','BedroomAbvGr']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                    random_state=7)

print(X_train.shape, y_train.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# define model architecture
model = Sequential()
model.add(Dense(6, activation="relu", name="layer1"))
model.add(Dense(500, activation="relu", name="layer2"))
model.add(Dense(250, activation="relu", name="layer3"))
model.add(Dense(250, activation="relu", name="layer4"))
model.add(Dense(500, activation="relu", name="layer5"))
model.add(layers.Dense(20, activation="relu", name="layer6"))
model.add(Dense(1, name="layer7"))
model.compile(optimizer='rmsprop', loss='mse')
# train model, generate weigts and apply gradient optimization
# can also check for validation loss using validation parameter
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.fit(X_train, y_train, epochs=200, 
            validation_data=(X_test, y_test),
            batch_size=32, callbacks=[callback])
# view model summary and network architecture
model.summary()
# obtain loss history
model.history.history
# define as dataframe
loss_history = pd.DataFrame(model.history.history)
# check loss graph
loss_history.plot()

# evaluating in-sample, and out-smaple error
model.evaluate(X_train, y_train)
model.evaluate(X_test, y_test)

# obtain model predictions
y_pred = model.predict(X_test)
# obtain pediction error
r2_score(y_test, y_pred)

#TODO: fix the plots
# visualize model prediction with true labels
predicted_labels = pd.Series(y_pred.reshape(len(y_pred),))

true_labels = pd.Series(y_test)

df = pd.concat([predicted_labels, true_labels], axis=1)
predicted_labels.plot()
plt.scatter(y_test, predicted_labels, data=df)
plt.legend()
plt.show()

# if model is amazing and you need to save it
model.save('amazing_predictor.h5')

# when wanting to load it
my_model = load_model('amazing_predictor.h5')