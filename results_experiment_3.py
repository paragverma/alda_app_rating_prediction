import datasets, preprocessing
import numpy as np
from doc2vecmodel import build_doc2vec, transform_df
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#datasets gives 3 variables
#dataset = Original dataset
#dataset2 = Reviews Dataset
#common_dataset = Inner Join of dataset and dataset2.
#By default the aggregation of all scores is the mean
#common_dataset = datasets.common_dataset
common_dataset = datasets.common_dataset

preprop_df = preprocessing.processDf(common_dataset)

vector_size = 5

models = []
models.append(LinearRegression())
models.append(KNeighborsRegressor(n_neighbors=20))
models.append(SVR(gamma='auto'))
models.append(RandomForestRegressor(n_estimators=200))

model_mse = {}
for model in models:
	model_mse[str(model).split("(")[0]] = []


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

X = 0
train_indices = 0
test_indices = 0
for fold in kf.split(np.arange(preprop_df.shape[0])):
	train_indices = fold[0]
	test_indices = fold[1]
	
	doc2vec_model = build_doc2vec(df = common_dataset,
					   df_reviews = datasets.dataset2,
					   train_indices = train_indices,
					   test_indices = test_indices,
					   vsize = vector_size)
	
	modified_df = transform_df(doc2vec_model = doc2vec_model,
							df_r = common_dataset.copy(deep=True),
							df_reviews = datasets.dataset2,
							vsize = vector_size)
	
	modified_df = preprocessing.processDf(modified_df)
	
	X, y = preprocessing.getNumpyXy(modified_df, 'Rating')
	
	X_train = X[train_indices]
	y_train = y[train_indices]
	X_test = X[test_indices]
	y_test = y[test_indices]
	
	for model in models:
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		mse = mean_squared_error(y_test, y_pred)
		model_mse[str(model).split("(")[0]].append(mse)

for model in models:
	name = str(model).split("(")[0]
	print(name, " : ", np.mean(model_mse[name]))