import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.core import Activation

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,\
    confusion_matrix
####################################################

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, 
                                                    test_size=0.10, 
                                                    random_state=7,
                                                    shuffle=True,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# label must be encoded prior to model fitting
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# KEras model
# input dim is the expected features
# units defines the first neuronal layer, we will ue 2x the feature

model = keras.Sequential(
                    [
                        keras.layers.Dense(100, activation='relu'),
                        keras.layers.Dense(50, activation='relu'),
                        keras.layers.Dense(3, activation='softmax')
                    ]
)

loss_function = keras.losses.CategoricalCrossentropy(
                    name="categorical_crossentropy",
)

model.compile(
                optimizer='adam', 
                loss=loss_function, 
                metrics=['accuracy']
)

model.fit(
            x=X_train_scaled, 
            y=y_train, 
            batch_size=10,
            epochs=100, 
            verbose=2,
            validation_data=(
                              X_test_scaled, 
                              y_test
                            )
)

model.summary()

# visualizing loss
# spike in validation loss indicates model over-fitting
model_loss = pd.DataFrame(model.history.history)
model_loss[['loss', 'val_loss']].plot()

# getting model predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes=np.argmax(y_pred, axis=1)

y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred_classes))
print(confusion_matrix(y_test, y_pred_classes))
