from operator import index
import numpy as np
from numpy.lib.histograms import histogram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hrt = pd.read_csv('heart_failure_clinical_records_dataset.csv')
#hrt1 = hrt.drop(['DEATH_EVENT','time'],axis=1, inplace=True)

#print(hrt1.columns)

#sns.set_style('whitegrid')
#sns.pairplot(hrt.drop(['anaemia','diabetes','high_blood_pressure','sex','smoking','DEATH_EVENT'], axis=1), hue='')
#plt.show() 

from sklearn.model_selection import train_test_split
X = hrt.drop(['DEATH_EVENT', 'time'], axis=1)
y = hrt['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

ln = LinearRegression()
ln.fit(X_train, y_train)

#print(ln.intercept_)

cf_df = pd.DataFrame(ln.coef_, X.columns)
cf_df.columns = ['Coef value']
print(cf_df)

pred = ln.predict(X_test)
#plt.scatter(y_test,pred)
#sns.distplot((y_test-pred),bins=50)
#plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,pred))
print('MSE:', metrics.mean_squared_error(y_test,pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,pred))) 
