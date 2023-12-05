# -*- coding: utf-8 -*-


import pandas as pd
import numpy as ny
import seaborn as sns
import statistics
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Importing the data file

Online_retail = pd.read_csv('online_shoppers_intention.csv',sep=',') 
Online_retail.head()
Online_retail_1 = pd.DataFrame(Online_retail)
# Exploratory Data Analysis
# Descriptive Statistica of data
Online_retail_1.describe()
# Check dimensions
Online_retail.shape
# Check data type of each feature
Online_retail.dtypes
# Data Processing 
# Data cleaning- missing values
# Checking for NA Values
data1 = Online_retail_1
data1.info()
monthly = data1['Month'].value_counts()
Month={'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'June':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
data1['Month']=data1['Month'].map(Month)
VisitorType={'Returning_Visitor':3, 'New_Visitor':2, 'Other':1}
data1['VisitorType']=data1['VisitorType'].map(VisitorType)
d={True:1,False:0}
data1['Weekend']=data1['Weekend'].map(d)
data1['Revenue']=data1['Revenue'].map(d)
# Data Integration - correlation analysis
cor = data1.corr()
sns.heatmap(data1.corr(), vmin=(-1), vmax=(1), annot=True)
# from correlation plot below features are removed 
data = data1.drop(['OperatingSystems','Browser','Region','TrafficType','Weekend'], axis = 1)
data = data1
# Logistic Regression

y = data['Revenue']
x = data.drop(['Revenue'], axis = 1)
scale_x = StandardScaler()
x1 = scale_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x1, y,stratify =y, test_size =  0.2, random_state = 0)
# K-fold
folds = range(2,31)
### evaluate the model using a given test condition
def evaluate_model(cv):
  # get the model
  model = LogisticRegression()
  # evaluate the model
  scores = cross_val_score(model, x1, y, scoring='accuracy', cv=cv, n_jobs=-1)
  # return scores
  return statistics.mean(scores), scores.min(), scores.max()

# evaluate each k value
for k in folds:
  # define the test condition
  cv = KFold(n_splits=k, shuffle=True, random_state=10)
  # record mean and min/max of each set of results
  k_mean, k_min, k_max = evaluate_model(cv)
  # report performance
  print('-> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logpred = logreg.predict(x_test)
print(classification_report(y_test,logpred))
log_acc = accuracy_score(y_test,logpred,normalize=True)
print('The Accuracy for test data using Logistic Regression is', log_acc)
#SVM
svm_mod = svm.SVC()
svm_mod.fit(x_train, y_train)
svm_pred = svm_mod.predict(x_test)
print(classification_report(y_test,svm_pred))
svm_acc = accuracy_score(y_test,svm_pred,normalize=True)
print('The Accuracy for test data using SVM is', svm_acc)
#KNN

for i in range(1, 14):
 print(i)
 knn = KNeighborsClassifier(n_neighbors = i)
 knn.fit(x_train, y_train) 
 knn_pred_i = knn.predict(x_test)
 print(classification_report(y_test,knn_pred_i))
 knn_acc = accuracy_score(y_test,knn_pred_i,normalize=True)
 print('The Accuracy for test data using KNN is', knn_acc)
#KNN 
knn1 = KNeighborsClassifier(n_neighbors = 11) 
knn1.fit(x_train, y_train) 
knn1_pred = knn1.predict(x_test)
print(classification_report(y_test,knn1_pred))
knn1_acc = accuracy_score(y_test,knn1_pred)
print('The Accuracy for test data using KNN is', knn1_acc)
# Naive Bayes
NB= GaussianNB()
NB.fit(x_train, y_train)
NB_pred = NB.predict(x_test)
print(classification_report(y_test,NB_pred))
NB_acc = accuracy_score(y_test,NB_pred,normalize=True)
print('The Accuracy for test data using Naive Bayes',NB_acc)


# Ramdom Forests 
RMC = RandomForestClassifier(n_estimators= 50, max_features=7 ,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

#Random Forest Hyperparameter Tuning - a

RMC = RandomForestClassifier(n_estimators=2000, max_features=7, random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features=17,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features=15,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features= 10,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features= 5,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features= 2,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 

RMC = RandomForestClassifier(n_estimators= 100, max_features=1,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
## #Random Forest Hyperparameter Tuning - b

RMC = RandomForestClassifier(n_estimators= 100, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
RMC = RandomForestClassifier(n_estimators= 10, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
RMC = RandomForestClassifier(n_estimators= 30, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
RMC = RandomForestClassifier(n_estimators= 50, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
RMC = RandomForestClassifier(n_estimators= 150, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
RMC = RandomForestClassifier(n_estimators= 500, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 
RMC = RandomForestClassifier(n_estimators= 1000, max_features= 7,random_state=12)
RMC.fit(x_train,y_train)
rmcy_pred = RMC.predict(x_test)
print(classification_report(y_test,rmcy_pred))
RMC_acc = accuracy_score(y_test,rmcy_pred,normalize=True)
print('The Accuracy for the test data using Random Forest is', RMC_acc) 


