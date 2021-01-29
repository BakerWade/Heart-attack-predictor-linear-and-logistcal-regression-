import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hrt = pd.read_csv('heart_failure_clinical_records_dataset.csv')
hrt = hrt.drop(['platelets','serum_sodium','time'],axis=1)

from sklearn.model_selection import train_test_split
X = hrt.drop('DEATH_EVENT',axis=1)
y = hrt['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train,y_train)

pred = log.predict(X_test)

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test,pred))
print('MAE:', metrics.mean_absolute_error(y_test,pred))
print('MSE:', metrics.mean_squared_error(y_test,pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,pred))) 




"""sns.set_style('whitegrid')
g = sns.FacetGrid(hrt,col='DEATH_EVENT',row='smoking',hue='sex')
g.map(plt.scatter,'age','serum_sodium').add_legend()
plt.show()"""

#diabetes
#anaemia
#high_blood_pressure
#smoking