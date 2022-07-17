import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Mapping Origin column
def preprocess_origin_col(df) :
    df['Origin'] = df['Origin'].map({1 : 'India',2 : 'USA', 3: 'Germany'})
    return df


# Adding custom attributes


acc_i = 4
cyl_i = 0
hp_i = 2


class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):
        self.acc_on_power = acc_on_power

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x):
        acc_on_cyl = x[:, acc_i] / x[:, cyl_i]
        if self.acc_on_power:
            acc_on_power = x[:, acc_i] / x[:, hp_i]
            return np.c_[x, acc_on_power, acc_on_cyl]
        return np.c_[x, acc_on_cyl]

def num_pipeline_transformer(data) :

    #pipeline for numerical attributes
    #imputing >> attribute adding >> scaling
    num_data = data.drop('Origin', axis = 1)

    num_pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('attr',CustomAttrAdder()),
         ('scale', StandardScaler())
         ])
    return num_pipeline, num_data

def pipeline_transformer(data) :
 #coloumn transformer to each column
    cat_data = ['Origin']
    num_pipeline , num_data = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, list(num_data)),
        ('ohe', OneHotEncoder(), cat_data)
        ])
    prepared_pipeline = full_pipeline.fit_transform(data)
    return prepared_pipeline


def predict_mpg(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    preprocess = preprocess_origin_col(df)
    final_prepare = pipeline_transformer(preprocess)
    pred = model.predict(final_prepare)
    return pred