import matplotlib as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics  import mean_squared_error

diabetes=datasets.load_diabetes()
diabetes_x=diabetes.data[:, np.newaxis,2]
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[:30]
diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[:30]
model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
predicted=model.predict(diabetes_x_test)
print("Mean squared error is : ",mean_squared_error(diabetes_y_test,predicted))
