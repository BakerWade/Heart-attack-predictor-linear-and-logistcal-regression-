import numpy as np
from numpy.lib.histograms import histogram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hrt = pd.read_csv('heart_failure_clinical_records_dataset.csv')
#hrt1 = hrt.drop(['DEATH_EVENT','time'],axis=1)

from sklearn.model_selection import train_test_split
X = hrt.drop(['DEATH_EVENT','time'], axis=1)
y = hrt['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(X_train, y_train)

pred = lg.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))