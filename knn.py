# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/159IvTLpLOMQAfyNXC7Mv8v1DCJiIBwRW
"""

# question 1 zoo

import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

zoo=pd.read_csv("Zoo.csv")
zoo

zoo.dtypes

zoo.shape

zoo.describe()

zoo.type.value_counts()

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
zoo["animal name"] = label_encoder.fit_transform(zoo["animal name"])

zoo.head()

array=zoo.values
X=array[:,1:17]
X

Y=array[:,-1]
Y

num_folds = 20
kfold = KFold(n_splits=20)

model = KNeighborsClassifier(n_neighbors=25)
results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())

## Grid Search for Algorithm

import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x=array[:,1:17]
x

y=array[:,-1]
y

n_neighbors = numpy.array(range(1,50))
param_grid = dict(n_neighbors=n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x, Y)

print(grid.best_score_)
print(grid.best_params_)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt 
# %matplotlib inline
k_range = range(1, 51)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, Y, cv=5)
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

## question 2 glass

import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

glass=pd.read_csv("glass.csv")
glass

glass.describe()

glass.shape

glass.dtypes

glass.Type.value_counts()

sns.scatterplot(glass['RI'],glass['Na'],hue=glass['Type'])

X = glass.iloc[:, 0:9]
Y = glass['Type']

X

Y

num_folds = 20
kfold = KFold(n_splits=20)

model = KNeighborsClassifier(n_neighbors=30)
results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())

## Grid Search for Algorithm

import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x= glass.iloc[:, 0:9]
y= glass['Type']

n_neighbors = numpy.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x, y)

print(grid.best_score_)
print(grid.best_params_)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt 
# %matplotlib inline
k_range = range(1, 41)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=5)
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

