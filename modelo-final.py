# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 20:20:53 2016

@author: rebli
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from keras.layers import Dense, Activation, Dropout, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#matplotlib.use('Agg')
from pylab import rcParams
rcParams['figure.figsize'] = 10,10



def prepardados(train,  test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    labels_cat = to_categorical(labels)
    classes = list(le.classes_)

    test_ids = test.id
    train_ids = train.id

    train = train.drop(['id', 'species'], axis=1)
    test = test.drop(['id'], axis=1)

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, labels_cat, classes, test_ids, test

def submicao(preds):
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()
    submission.to_csv('submit.csv', index=False)

	
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')	
train, labels, classes, test_ids, test = prepardados(train, test)


model = Sequential()
model.add(Dense(1024, input_dim=192))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(99))
model.add(Activation("softmax"))



model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
print("Iniciando....")
history = model.fit(train, labels, nb_epoch=130, batch_size=128,validation_split=0.2)
print("Finalizando.....")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valida'], loc='upper left')
plt.savefig("figura1.png")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valida'], loc='upper left')
plt.savefig("figura2.png")

preds = model.predict_proba(test)
submicao(preds)




