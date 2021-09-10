# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:23:31 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON/Classification")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Data.csv")
dataset.columns
x=dataset[['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape', 'Marginal Adhesion','Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli', 'Mitoses']]
y=dataset[['Class']]

dataset.isnull().any()
dataset.apply(lambda x:x.isnull().sum()/len(x)*100)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)

#Logistic Reression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)

y_pred=lr.predict(xtest)
print(np.concatenate((y_pred.reshape(len(y_pred),1),ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import confusion_matrix,r2_score
cm=confusion_matrix(ytest,y_pred)
r2_score(ytest,y_pred)

from sklearn.svm import SVC
sv=SVC(kernel='rbf', random_state=0)
sv.fit(xtrain,ytrain)

y_pred2=sv.predict(xtest)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1),ytest.values.reshape(len(ytest),1)),1))
cm2=confusion_matrix(ytest,y_pred2)
r2_score(ytest,y_pred2)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain,ytrain)

y_pred3=knn.predict(xtest)
print(np.concatenate((y_pred3.reshape(len(y_pred3),1),ytest.values.reshape(len(ytest),1)),1))
cm3=confusion_matrix(ytest,y_pred3)
r2_score(ytest,y_pred3)

from sklearn.tree import DecisionTreeClassifier
dec_tree=DecisionTreeClassifier(criterion='entropy', random_state=0)
dec_tree.fit(xtrain,ytrain)

y_pred4=dec_tree.predict(xtest)
print(np.concatenate((y_pred4.reshape(len(y_pred4),1),ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(ytest,y_pred4)
r2_score(ytest,y_pred4)

from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=10, random_state=0)
rc.fit(xtrain,ytrain)

y_pred5=rc.predict(xtest)
print(np.concatenate((y_pred5.reshape(len(y_pred5),1),ytest.values.reshape(len(ytest),1)),1))

cm4=confusion_matrix(ytest,y_pred4)
r2_score(ytest,y_pred4)

