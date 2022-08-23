# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 17:07:53 2022

@author: Rakesh
"""

##Coding done on Sypder#
##importing libraries#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##loading data set#
fedex = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/FedEx/fedex.csv')

##checking headder and footer dataset#
fedex.head(10)
fedex.tail(10)

fedex.describe()

##let import some more libraries #
from sklearn.preprocessing import LabelEncoder #for converting categorical to numerical#
from sklearn.compose import ColumnTransformer #to conversion of column#
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split ##train and testing#

##removing Year column #
fedex= fedex.drop('Year' , axis=1)

##checking null values #
fedex.isna().sum()

##dropping#
fedex1= fedex.dropna()
fedex1

##Label encoding#
LE= LabelEncoder()
fedex1['Carrier_Name']=LE.fit_transform(fedex1['Carrier_Name'])
fedex1['Source']=LE.fit_transform(fedex1['Source'])
fedex1['Destination']=LE.fit_transform(fedex1['Destination'])

##Normalization##
mm= MinMaxScaler()
fedex2=fedex1.to_numpy()
fedex2= mm.fit_transform(fedex2)
fedex1= pd.DataFrame(fedex2,columns=fedex1.columns)
fedex1

##Decision Tree#
predictors= fedex1.loc[:, fedex1.columns!='Delivery_Status']
type(predictors)

target = fedex1['Delivery_Status']
type(target)

## Spliting Test train dataset#
x_train, x_test, y_train, y_test= train_test_split(predictors,target, test_size= 0.2, random_state=0)

from sklearn import tree
regtree = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 5)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

from sklearn.metrics import mean_squared_error , r2_score

##error on test dataset#
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

##error on train dataset#
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)

####bagging Classifier#
from sklearn import tree
clftree=tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier

bag_clf= BaggingClassifier(base_estimator=clftree, n_estimators=100,
                           bootstrap=True, n_jobs=1 , random_state=42)
bag_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

###ADA boost#
from sklearn.ensemble import AdaBoostClassifier
ada_clf= AdaBoostClassifier(learning_rate=0.02,n_estimators=500)
ada_clf.fit(x_train,y_train)
# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

##Gradient Boost#
from sklearn.ensemble import GradientBoostingClassifier
boost_clf= GradientBoostingClassifier()
boost_clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix    
confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))

accuracy_score(y_train, boost_clf.predict(x_train))

##Xgradient Boost##
import xgboost as xgb
xgb_clf= xgb.XGBClassifier(n_estimator=500 , learning_rate=0.1 , random_state=42)
xgb_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

####K-nearest Neighbour#
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test, knn.predict(x_test))
accuracy_score(y_test, knn.predict(x_test))

confusion_matrix(y_train, knn.predict(x_train))
accuracy_score(y_train, knn.predict(x_train))

###Random Forest Classifier#
from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier(n_estimators=100, oob_score=True , random_state=101 , max_features=None , max_depth=1 , min_samples_leaf=5)
rdf.fit(x_train, y_train)

y_pred= rdf.predict(x_test)
from sklearn.metrics import accuracy_score ,confusion_matrix
    
confusion_matrix(y_test, rdf.predict(x_test))
accuracy_score(y_test, rdf.predict(x_test))

########SGD Classifier###################
from sklearn.linear_model import SGDClassifier 
sgd= SGDClassifier(loss='modified_huber', shuffle= True , random_state=101)
sgd.fit(x_train , y_train)

from sklearn.metrics import accuracy_score ,confusion_matrix
##on Test data#
confusion_matrix(y_test, sgd.predict(x_test))
accuracy_score(y_test, sgd.predict(x_test))

##on train data##
confusion_matrix(y_train, sgd.predict(x_train))
accuracy_score(y_train, sgd.predict(x_train))

###Conslusion is data is 90% accurate ###






