import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

data = load_breast_cancer()
X, y = load_breast_cancer(return_X_y=True)

df = pd.DataFrame(data.data, columns=[data.feature_names])
df['target'] = data.target

# drop multilevel indexing
cols = []
for i in df.columns:
    cols.append(i[0])

df.columns = cols

#basic frame info
df.columns
df.info()
df.describe().transpose()
# target count
sns.countplot(target)

# frame correlation to target using bar plots
df.corr()['target'].sort_values().plot(kind='bar')
# dropping last column which is target
df.corr()['target'][:-1].sort_values().plot(kind='bar')

#heatmap
sns.heatmap(df.corr())

# setup figure size
plt.figure(figsize=(8, 10))
# set column correlation to target - in this case 'Outcome
heatmap = sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), 
                      vmin=-1, vmax=1, annot=True, cmap='magma', fmt='0.2f')
# title, font and padding
heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16);


# normalizing data, splitting and 
sc = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    shuffle=True, stratify=df.target, random_state=7)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# DNN MODEL
model = Sequential()

model.add(Dense(30, activation='relu', name='input_layer'))
model.add(Dense(15, activation='relu', name='hidden_layer1'))
model.add(Dense(5, activation='relu', name='hidden_layer2'))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.compile(optimizer='SGD', loss='binary_crossentropy', metrics='accuracy')

model.fit(X_train, y_train, batch_size=3, epochs=500, 
            validation_data=(X_test, y_test))

# visualizing loss
# spike in validation loss indicates model over-fitting
model_loss = pd.DataFrame(model.history.history)
model_loss[['loss', 'val_loss']].plot()

###########################################
####prevent over-fitign with early stopping
###########################################
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

model = Sequential()

model.add(Dense(30, activation='relu', name='input_layer'))
model.add(Dense(15, activation='relu', name='hidden_layer1'))
model.add(Dense(5, activation='relu', name='hidden_layer2'))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.compile(optimizer='SGD', loss='binary_crossentropy', metrics='accuracy')

model.fit(X_train, y_train, batch_size=3, epochs=500, 
            validation_data=(X_test, y_test), callbacks=[stop])

model_loss2 = pd.DataFrame(model.history.history)
# visualize model output without over-fitting
model_loss2[['loss', 'val_loss']].plot()

#####################################################
####prevent over-fitign with early stopping + DropOut
#####################################################
model = Sequential()

model.add(Dense(30, activation='relu', name='input_layer'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu', name='hidden_layer1'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

model.fit(X_train, y_train, batch_size=3, epochs=500, 
            validation_data=(X_test, y_test), callbacks=[stop])

model_loss3 = pd.DataFrame(model.history.history)
# visualize model output without over-fitting
model_loss3[['loss', 'val_loss']].plot()

#PREDICTING model output
y_pred = model.predict(X_test)
y_pred_classes=np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes))
print(confusion_matrix(y_test, y_pred_classes))

##########################################################
#########Model visualization with Tensorboard############
#########################################################
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d--%H%M')
log_dir = "logs/fit/"# + datetime.now().strftime("%Y%m%d-%H%M%S")

my_board = TensorBoard(log_dir=log_dir, 
                        histogram_freq=5,
                        write_graph=True,
                        write_images=True,
                        update_freq='epoch',
                        profile_batch=3)

model = Sequential()
model.add(Dense(30, activation='relu', name='input_layer'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu', name='hidden_layer1'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', name='output_layer'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

model.fit(X_train, y_train, batch_size=3, epochs=500, 
            validation_data=(X_test, y_test), callbacks=[stop, my_board])

# Run tesorboard at local browser 
# run in command line
        # tensorboard --logdir keras_api\logs\fit
        # access localhost 