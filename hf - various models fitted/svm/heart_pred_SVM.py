import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

hartfail= pd.read_csv('heart_failure_clinical_records_dataset.csv')

from sklearn.model_selection import train_test_split

X= hartfail.drop(['DEATH_EVENT'],axis=1)
y = hartfail['DEATH_EVENT']

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)

prediction= model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

#print(confusion_matrix(y_test,prediction))
#print(classification_report(y_test,prediction))

from sklearn.model_selection import GridSearchCV
param= {'C': [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param, refit=True, verbose= 3)
grid.fit(X_train, y_train)

pred = grid.predict(X_test)

print(grid.best_params_)
print(grid.best_estimator_)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))