import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pipeline import prepped_train_func
import joblib

prepped_train = prepped_train_func("housing.csv")
df_housing = pd.read_csv("housing.csv")
df_housing.insert(0, "bedrooms_per_house", df_housing["total_bedrooms"] / df_housing["total_rooms"])

train_housing, test_housing = train_test_split(df_housing, test_size=0.2, random_state=75)

# forest_reg = RandomForestRegressor(random_state = 32,max_depth=10)
# scoring = cross_val_score(forest_reg,prepped_train,train_housing["median_house_value"],cv = 5 ,scoring='r2')

# forest_reg.fit(prepped_train, train_housing["median_house_value"])
# train_predict = forest_reg.predict(prepped_train)

# train_mse = mean_squared_error(train_housing["median_house_value"], train_predict) 
# print(scoring)
# print(train_mse)

# params = {"n_estimators": [100,200,300],
#     'max_depth': [10, 20, 30]}

# search = GridSearchCV(estimator = forest_reg,param_grid = params,cv=5)

# search.fit(prepped_train,train_housing["median_house_value"])
# print(search.best_params_)
# print(search.best_score_)

best_reg = RandomForestRegressor(random_state=32,max_depth=20,n_estimators=200)
scoring = cross_val_score(best_reg,prepped_train,train_housing["median_house_value"],cv = 5 ,scoring='r2')

best_reg.fit(prepped_train, train_housing["median_house_value"])
train_predict = best_reg.predict(prepped_train)

train_mse = mean_squared_error(train_housing["median_house_value"], train_predict) 
print(scoring)
print(train_mse)

joblib.dump(best_reg, 'main.joblib')
