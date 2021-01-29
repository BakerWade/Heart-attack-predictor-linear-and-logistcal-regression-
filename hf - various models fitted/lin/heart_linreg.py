import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hrt = pd.read_csv('heart_failure_clinical_records_dataset.csv')
hrt = hrt.drop(['serum_sodium','platelets','time'],axis=1)

#print(hrt1.columns)

sns.set_theme(style='white')
#sns.pairplot(hrt)
#sns.countplot(x=hrt['DEATH_EVENT'])
#g = sns.JointGrid(x='age', y='creatinine_phosphokinase', data=hrt, space=0)
#g.plot_joint(sns.kdeplot, fill=True, thresh=0, levels=45, cmap="icefire")
#g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=50, kde=True)

#sns.jointplot(data=hrt, x='age', y='creatinine_phosphokinase', hue='smoking', kind='kde')
#plt.show() 

from sklearn.model_selection import train_test_split
X = hrt.drop('DEATH_EVENT', axis=1)
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
plt.show()

from sklearn import metrics


print('MAE:', metrics.mean_absolute_error(y_test,pred))
print('MSE:', metrics.mean_squared_error(y_test,pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,pred))) 

