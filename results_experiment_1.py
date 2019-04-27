import datasets, preprocessing
import numpy as np
#datasets gives 3 variables
#dataset = Original dataset
#dataset2 = Reviews Dataset
#common_dataset = Inner Join of dataset and dataset2.
#By default the aggregation of all scores is the mean
#common_dataset = datasets.common_dataset
common_dataset = datasets.common_dataset
common_dataset = preprocessing.processDf(common_dataset)

common_dataset = common_dataset.drop(['Sentiment_Polarity', 'Sentiment_Subjectivity'], axis=1)


X, y = preprocessing.getNumpyXy(common_dataset, 'Rating')


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
model = LinearRegression()
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Average MSE Linear = " + str(-np.mean(cv_rmse)))


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=20)
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Average MSE KNNRegressor = " + str(-np.mean(cv_rmse)))

from sklearn.svm import SVR
model = SVR(gamma='auto')
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Average MSE SVR = " + str(-np.mean(cv_rmse)))


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200)
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Average MSE RFR = " + str(-np.mean(cv_rmse)))