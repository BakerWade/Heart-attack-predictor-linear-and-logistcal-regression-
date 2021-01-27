import numpy as np
import pandas as pd

hartfail= pd.read_csv('heart_failure_clinical_records_dataset.csv')

X = hartfail.drop(['DEATH_EVENT'], axis=1)
y = hartfail['DEATH_EVENT']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

#print(classification_report(y_test,prediction))
#print(confusion_matrix(y_test,prediction))

from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier(n_estimators=1000)

ran.fit(X_train, y_train)
pred = ran.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))