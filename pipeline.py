import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def prepped_train_func(csv_path):
    df_housing = pd.read_csv(csv_path)
    #Not sure if you want us to add this column for every data we would
    #hypothetically work with
    df_housing.insert(0, "bedrooms_per_house", df_housing["total_bedrooms"] / df_housing["total_rooms"])

    # 
    train_housing, test_housing = train_test_split(df_housing, test_size=0.2, random_state=75)

    num_attr = df_housing.iloc[:,:-1]
    cat_attr = ["ocean_proximity"]

    num_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(sparse_output=False))
    ])

    preprocess = ColumnTransformer(transformers=[
        ('num_pipeline', num_pipeline, num_attr.columns),
        ('cat_pipeline', cat_pipeline, cat_attr)
    ])

    prepped_train = preprocess.fit_transform(train_housing)

    return prepped_train