# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:17:04 2020

@author: s133016
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\Spotify2019\top50data.txt', delimiter = ',',encoding='ISO-8859-1')

df.rename(columns = {'Track.Name':'track_name', 
                     'Artist.Name':'artist_name',
                     'Beats.Per.Minute':'beats_per_minute',
                     'Loudness..dB..':'Loudness(dB)',
                     'Valence.':'Valence',
                     'Length.':'Length',
                     'Acousticness..':'Acousticness',
                     'Speechiness.':'Speechiness'},
inplace = True
)

df_new = df.drop(['track_name', 'Unnamed: 0'], axis = 1, inplace=False)

X =df_new.drop(['Popularity'], axis = 1)
y = df_new['Popularity']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

numerical_features = [col for col in X_train.columns if X_train[col].dtypes!= 'O']

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X[self.feature_names].values
    
class new_LabelBina(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()
    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self
    def transform(self, X):
        return self.encoder.transform(X)
   
class new_Label_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, X, y = None):
        self.encoder.fit(X)
        return self
    def transform(self, X):
        return self.encoder.transform(X).reshape(-1,1)

numerical_pipeline = Pipeline(steps = [('selector',
                                        FeatureSelector(numerical_features)),
    ('std_scalar', StandardScaler())])
    
artist_pipeline = Pipeline(steps = [('selector', FeatureSelector(['artist_name'])),
                                         ('label_binarizer', new_LabelBina()),
                                         ])
genre_pipeline = Pipeline(steps = [('selector', FeatureSelector(['Genre'])),
                                   ('label_binarizer2', new_LabelBina()),
                                   ])
    
full_pipeline = FeatureUnion(transformer_list = [('numerical_pipeline',
                                                   numerical_pipeline), ('artist_pipeline', artist_pipeline),
                                                   ('genre_pipeline', genre_pipeline)])
    
df_prepared = full_pipeline.fit_transform(X_train)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear = LinearRegression()
scale_output = StandardScaler()
y_new = scale_output.fit_transform(np.array(y_train).reshape(-1,1))

linear.fit(df_prepared, y_new)
test_data = full_pipeline.transform(X_test)
y_predict = linear.predict(test_data)

y_new = scale_output.transform(np.array(y_test).reshape(-1,1))

MSE = np.sqrt(mean_squared_error(y_new, y_predict))
print(MSE)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB

knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(df_prepared, y_train)
y_predict = knn.predict(test_data)
error = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(df_prepared, y_train)
    pred_i = knn.predict(test_data)
    error.append(np.mean(pred_i!=y_test))
    

import matplotlib.pyplot as plt    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(range(1,30), error, color = 'black',
        marker = 'o', markerfacecolor = 'cyan',
        markersize = 10)
ax.set_xlabel('K Value')
ax.set_ylabel('Mean Error')
ax.set_title('Error Rate K Value')