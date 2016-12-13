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
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Activation, Dropout, Flatten, Merge
from keras.layers.normalization import BatchNormalization
#matplotlib.use('Agg')
from pylab import rcParams
rcParams['figure.figsize'] = 10,10



def prepardados(train,y_label):
    le = LabelEncoder().fit(y_label)
    labels = le.transform(y_label)
    labels_cat = to_categorical(labels)
    classes = list(le.classes_)

   

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    

    return train, labels_cat, classes

	
	
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_label = train['species']

data = train.drop(['species','id'],axis=1)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
data = data.as_matrix()

for train1, test1 in kfold.split(data, y_label):
  data_train = data[train1]
  train_labels = y_label[train1]
  train_10, labels_10, classes = prepardados(data_train,train_labels)
  data_test = data[test1]
  y_test = y_label[test1]
  test_20, labels_20, classes1 = prepardados(data_test,y_test)
  
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
  model.fit(train_10, labels_10, nb_epoch=130, batch_size=128)
  scores = model.evaluate(test_20, labels_20)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
  print("Finalizando.....")

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

np.savetxt("scores.txt", cvscores, delimiter=',')





