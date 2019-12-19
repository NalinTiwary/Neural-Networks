from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def categorical_to_numpy(labels_in):
  labels = []
  for label in labels_in:
    if label == 'dog':
      labels.append(np.array([1, 0]))
    else:
      labels.append(np.array([0, 1]))
  return np.array(labels)


def load_data():
  import gdown
  gdown.download('https://drive.google.com/uc?id=1-BjeqccJdLiBA6PnNinmXSQ6w5BluLem','cifar_data','True'); # dogs v road;

  import pickle
  data_dict = pickle.load(open( "cifar_data", "rb" ));

  data   = data_dict['data']
  labels = data_dict['labels']

  return data, labels

data, labels = load_data()
inputs_train,inputs_test,labels_train,labels_test=train_test_split(data, labels, test_size=0.2, random_state=101)
nnet = MLPClassifier(hidden_layer_sizes=(10), max_iter= 10000)
nnet.fit(inputs_train, labels_train)

predictions = nnet.predict(inputs_test)

print("MLP Testing Set Score:")
print(accuracy_score(labels_test, predictions)*100)
