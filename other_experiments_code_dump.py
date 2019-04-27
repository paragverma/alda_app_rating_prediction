import datasets, preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#datasets gives 3 variables
#dataset = Original dataset
#dataset2 = Reviews Dataset
#common_dataset = Inner Join of dataset and dataset2.
#By default the aggregation of all scores is the mean
#common_dataset = datasets.common_dataset
common_dataset = datasets.common_dataset
common_dataset = preprocessing.kpreprocessing(common_dataset)

common_dataset = common_dataset.drop(['Sentiment_Polarity', 'Sentiment_Subjectivity'], axis=1)


X, y = preprocessing.getNumpyXy(common_dataset, 'Rating')

X_train, y_train = preprocessing.getNumpyXy(common_dataset_train, 'Rating')
X_test, y_test = preprocessing.getNumpyXy(common_dataset_test, 'Rating')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
print(mean_squared_error(y_test, y_pred))

X = X[:, cols]
scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1,1))

scaler = StandardScaler()
X = scaler.fit_transform(X)
pc = PCA()
pc.fit(X=X, y=y)

components = pc.components_
ev = pc.explained_variance_ratio_


print(np.mean(cv_rmse))


from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
model = LinearRegression()
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=20)
cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
model = KNeighborsRegressor(n_neighbors=20)
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
#0.876690923783342
#without - 0.9787753097843088



from multiisotonic import MultiIsotonicRegressor
model = MultiIsotonicRegressor()
cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
model = MultiIsotonicRegressor()
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')



from sklearn.svm import SVR
model = SVR()
cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
#Full, no scale, kpreprop -> [0.96590026, 0.9687558 , 0.8596631 , 0.7944252 , 0.81395299]
model = SVR()
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
#Full, no scale, kpreprop -> [-0.1187574 , -0.11917905, -0.26833002, -0.30119203, -0.26419876]
#0.8798234452473068
#0.8798293821363465
#w 0.9771132784693768

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200)

cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
#Full, no scale, kpreprop -> [0.96590026, 0.9687558 , 0.8596631 , 0.7944252 , 0.81395299]
model = RandomForestRegressor(n_estimators=200)
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')


from gplearn.genetic import SymbolicRegressor
model = SymbolicRegressor(population_size=100, generations=50)
cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
#Full, no scale, kpreprop -> [0.96590026, 0.9687558 , 0.8596631 , 0.7944252 , 0.81395299]
model = SymbolicRegressor()
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 15):
  print(i)
  cluster = KMeans(n_clusters=i)
  cluster.fit(X)
  inertia.append(cluster.inertia_)

#inertia = inertia[10:]
plt.plot(np.array([i + 1 for i in range(len(inertia))]), np.array(inertia))




from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

kmeans = KMeans(n_clusters=3, random_state=1)
y_clust_pred = kmeans.fit_predict(X_train).reshape(-1, 1)
X_clus = np.concatenate((X_train, y_train.reshape(-1, 1)), axis = 1)
X_clus = np.append(X_clus, y_clust_pred, axis = 1)
#X + y + y_clust_pred

splits = [X_clus[X_clus[:,-1]==k] for k in np.unique(X_clus[:,-1])]

classifiers = []

from sklearn.svm import SVR
for xset in splits:
  model = SVR()
  model.fit(xset[:, :-3], xset[:, -2])
  classifiers.append(model)



y_pred_clus = kmeans.predict(X_test).reshape(-1, 1)
X_test_clus = np.append(X_test, y_test.reshape(-1, 1), axis = 1)
X_test_clus = np.append(X_test_clus, y_pred_clus, axis = 1)

test_splits = [X_test_clus[X_test_clus[:,-1]==k] for k in np.unique(X_test_clus[:,-1])]

y_test_actual = []
y_pred = []

for i in range(len(test_splits)):
    y_test_actual += list(test_splits[i].T[test_splits[i].shape[1]- 2])
    y_pred += list(classifiers[i].predict(test_splits[i][:, :-3]))

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

recall_b_b = recall_score(y_test_actual, y_pred, average = 'weighted')
prec_b_b = precision_score(y_test_actual, y_pred, average = 'weighted')
f1_b_b = f1_score(y_test_actual, y_pred, average = 'weighted')
acc_b_b = accuracy_score(y_test_actual, y_pred)