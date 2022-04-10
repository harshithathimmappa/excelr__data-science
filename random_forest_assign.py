# -*- coding: utf-8 -*-
"""random forest assign.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RnVrnutMtuIEr8DpMtmvbr4YXHrn6Gi9
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression

company=pd.read_csv("Company_Data (1).csv")
company

company.dtypes

company.shape

company.describe()

company.Sales.mean()

company["sales"]="<-7.5"

company.loc[company["Sales"]<=7.5,"sales"]="Low Sales"
company.loc[company["Sales"]>=7.5,"sales"]="High Sales"

company=company.drop(["Sales"],axis=1)

label_encoder = preprocessing.LabelEncoder()
company['sales']= label_encoder.fit_transform(company['sales']) 
company['ShelveLoc']= label_encoder.fit_transform(company['ShelveLoc']) 
company['Urban']= label_encoder.fit_transform(company['Urban']) 
company['US']= label_encoder.fit_transform(company['US'])

company

x=company.iloc[:,0:10]
y=company['sales']

pd.set_option("display.max_rows", None)

x

y

company.sales.unique()

company.sales.value_counts()

colnames = list(company.columns)
colnames

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

"""## bulding model using random forest"""

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

num_trees = 150
max_features = 4
kfold = KFold(n_splits=10)

model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features,criterion='entropy')
results = cross_val_score(model, x, y, cv=kfold)

print(results.mean())

num_trees_1 = 600
max_features = 4
kfold = KFold(n_splits=10)

model_1 = RandomForestClassifier(n_estimators=num_trees_1, max_features=max_features,criterion='entropy')
results_1 = cross_val_score(model, x, y, cv=kfold)

print(results_1.mean())

## ridge regression

from sklearn.linear_model import Ridge 

# Train the model 
ridgeR = Ridge(alpha = 1) 
ridgeR.fit(x_train, y_train) 
y_pred = ridgeR.predict(x_test)

# calculate mean square error 
mean_squared_error_ridge = np.mean((y_pred - y_test)**2) 
print(mean_squared_error_ridge)

# get ridge coefficient and print them 
ridge_coefficient = pd.DataFrame() 
ridge_coefficient["Columns"]= x_train.columns 
ridge_coefficient['Coefficient Estimate'] = pd.Series(ridgeR.coef_) 
print(ridge_coefficient)

## question 2 fraud

import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

fraud=pd.read_csv("Fraud_check (1).csv")
fraud.head(10)

fraud.dtypes

fraud.shape

fraud.describe()

fraud["tax_income"]="<=30000"

fraud.loc[fraud["Taxable.Income"]>=30000,"tax_income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"tax_income"]="Risky"

fraud["tax_income"].unique()

fraud.head(10)

fraud=fraud.drop(["Taxable.Income"],axis=1)

fraud.rename(columns={"Undergrad":"UG","Marital.Status":"Marital","City.Population":"Population","Work.Experience":"exp"},inplace=True)

fraud.head(10)

label_encoder = preprocessing.LabelEncoder() 
fraud['UG']= label_encoder.fit_transform(fraud['UG']) 
fraud['tax_income']=label_encoder.fit_transform(fraud['tax_income'])
fraud['Urban']=label_encoder.fit_transform(fraud['Urban'])
fraud['Marital']=label_encoder.fit_transform(fraud['Marital'])

fraud.head(10)

fraud.dtypes

x=fraud.iloc[:,0:5]
y=fraud['tax_income']

x

y

fraud.tax_income.value_counts()

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

num_trees = 500
max_features = 3
kfold = KFold(n_splits=10)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features,criterion = 'gini')
results = cross_val_score(model, x, y, cv=kfold)

print(results.mean())

