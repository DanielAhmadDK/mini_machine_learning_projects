# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:47:22 2020

@author: s133016
"""

import pandas as pd
import numpy as np
from IPython.display import display
df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\heart_stroke\dataset.csv')

df_sample = df.sample(n = 1000)

def analyze(df_sample):
    print('The shape of the dataset is:', df_sample.shape[0])
    print('The variables of the dataset are:', df_sample.shape[1])
    print('-------------------------------')
    print('The columns are: \n')
    print(df_sample.columns)
    print('------------------->')
    print('The data type is: \n')
    print(df_sample.dtypes)
    print('------------------->')
    c = df_sample.isnull().sum()
    print('Unique values:', df_sample.apply(lambda x: [x.nunique()]))
    print(c[c>0])
analyze(df_sample)
    

df_new = df_sample[['Year', 'LocationAbbr','LocationDesc','Topic','Indicator',
                        'Data_Value', 'Data_Value_Alt',
                        'Confidence_Limit_Low','Confidence_Limit_High','Break_Out_Category',
                        'Break_out','CategoryID','TopicID',
                        'IndicatorID', 'Data_Value_TypeID',
                        'BreakOutID','LocationID','GeoLocation']]


def detailed_analysis(df_new, pred=None):
  obs = df_new.shape[0]
  types = df_new.dtypes
  counts = df_new.apply(lambda x: x.count())
  uniques = df_new.apply(lambda x: [x.unique()])
  nulls = df_new.apply(lambda x: x.isnull().sum())
  distincts = df_new.apply(lambda x: x.unique().shape[0])
  missing_ratio = (df_new.isnull().sum() / obs) * 100
  skewness = df_new.skew()
  kurtosis = df_new.kurt()
  print('Data shape:', df_new.shape)

  if pred is None:
    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)
  else:
    corr = df_new.corr()[pred]
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis, corr], axis=1, sort=False)
    corr_col = 'corr ' + pred
    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis', corr_col]

  details.columns = cols
  dtypes = details.types.value_counts()
  print('____________________________\nData types:\n', dtypes)
  print('____________________________')
  return details

details = detailed_analysis(df_new)

from sklearn.model_selection import train_test_split,cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.30)

numerical_cols = [col for col in df_new.columns if df_new[col].dtypes != 'O']
categorical_cols = [col for col in df_new.columns if df_new[col].dtypes == 'O']


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
    
    
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])