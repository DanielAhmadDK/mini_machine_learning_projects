# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:12:42 2020

@author: s133016
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\Spotify2019\top50data.txt', delimiter = ',',encoding='ISO-8859-1')

df.head()

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

df.head()


#check for missing values

df.isnull().sum()
df.fillna(0)

print(df.dtypes)

def data_process(df):
    print('The shape of the data is:', df.shape[0])
    print('The variables are:', df.shape[1])
    print('------------------->')
    print('The columns are: \n')
    print(df.columns)
    print('------------------->')
    print('The data type is: \n')
    print(df.dtypes)
    print('------------------->')
    c = df.isnull().sum()
    print(c[c>0])
data_process(df)


#Calculating the numer of songs of each genre
print(type(df['Genre']))
popular_genre = df.groupby('Genre').size().sort_values(ascending = False)
print(popular_genre)
genre_list = df['Genre'].values.tolist()
genre_list.sort()

#calculating the number of songs by each artist
print(df.groupby('artist_name').size())
popular_artist = df.groupby('artist_name').size().sort_values(ascending = False)
print(popular_artist)
artist_list = df['artist_name'].values.tolist()

df.isnull().sum()
df.fillna(0)


#set decimal precision
pd.set_option('precision', 2)
df.describe()


#finding out the skew for each attribute

skew = df.skew()
print(skew)
#from the output we can see at Liveness as a value of 2.2
#which means it is highly skewed to the right
#Popularity is highly skewed to the left with -1.5
#Danceability, speechiness has values -1.38 and 1.38
#removing the skew by using the boxcox transformations
from scipy import stats

transform = np.asarray(df[['Liveness']].values)
df_transform = stats.boxcox(transform)[0]

#plotting a histogram to show the difference

import matplotlib.pyplot as plt
plt.figure()
plt.hist(df['Liveness'], bins = 10)
plt.show()
plt.figure()
plt.hist(df_transform, bins = 10)
plt.show()


#transform popularity skew = -1.50

transform1 = np.asarray(df[['Popularity']].values)
df_transform1 = stats.boxcox(transform)[0]

import seaborn as sns
plt.figure()
sns.distplot(df['Popularity'], bins = 10, kde =True,
             kde_kws = {'color':'r', 'lw':2,
                        'label':'KDE'}, color = 'yellow')

plt.figure()
sns.distplot(df_transform1, bins = 10, kde =True,
             kde_kws = {'color':'k', 'lw':2,
                        'label':'KDE'}, color = 'black')

    
pd.set_option('display.width', 100)
pd.set_option('precision', 2)
correlation = df.corr(method = 'spearman')
print(correlation)

plt.figure(figsize=(10,10))
plt.title('Correlation Heatmap')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white',
                 vmin = -1, vmax = 1, center = 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
ax.set_ylim(len(correlation)-1, -1)           
plt.show()


fig, ax = plt.subplots(figsize = (30,12))
length = np.arange(len(popular_genre))
plt.barh(length, popular_genre, color = 'green', edgecolor = 'black', alpha = 0.7)
plt.yticks(length, genre_list)
plt.title('Most popular genre', fontsize = 18)
plt.ylabel('Genre', fontsize = 16)
plt.xlabel('Number of songs', fontsize = 16)
plt.show()

fig, ax = plt.subplots(figsize = (12,12))
length = np.arange(len(popular_artist))
plt.barh(length, popular_artist, color = 'red', edgecolor = 'k', alpha = 0.7)
plt.yticks(length, artist_list)
plt.title('Most popular artists', fontsize = 18)
plt.ylabel('Artists', fontsize = 16)
plt.xlabel('Number of songs', fontsize = 16)

sns.catplot(y = 'Genre', kind = 'count',
            palette = 'pastel', edgecolor = '.6',
            data = df)

#analysing the relationship between energy and loudness
#correlation is 0.64

fig = plt.subplots(figsize = (10,10))
sns.regplot(x = 'Energy', y = 'Loudness(dB)', data = df,
            color = 'black')

fig = plt.subplots(figsize = (10,10))
plt.title('Dependence between energy and popularity')
sns.regplot(x = 'Energy', y = 'Popularity', ci = None, data=df)
sns.kdeplot(df.Energy, df.Popularity)

import squarify as sq
plt.figure(figsize = (14, 8))
sq.plot(sizes = df.Genre.value_counts(), label = df['Genre'].unique(), alpha =0.8)
plt.axis('off')
plt.show()

labels = df.artist_name.value_counts().index
sizes = df.artist_name.value_counts().values
colors = ['red','yellowgreen','lightcoral','lightskyblue',
          'cyan','green','black','yellow']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels = labels, colors = colors)
autopct = ('%1.1f%%')
plt.axis('equal')
plt.show()


#now the machine learning starts

x = df.loc[:, ['Energy','Danceability','Length','Loudness(dB)','Acousticness']].values
y = df.loc[:,'Popularity'].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

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



#Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

#Difference between actual and prediction
y_pred = regressor.predict(X_test)
df_output = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(df_output)

print('Mean absolute error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean squared error:', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#from sklearn.metrics import accuracy_score

#cross validation score
x = df.loc[:, ['Energy','Danceability']].values
y = df.loc[:, 'Popularity'].values

regressor = LinearRegression()
mse = cross_val_score(regressor, X_train, y_train,
                      scoring = 'neg_mean_squared_error',
                      cv = 5)
mse_mean = np.mean(mse)
print(mse_mean)
diff = metrics.mean_squared_error(y_test, y_pred)-abs(mse_mean)
print(diff)

#try another approach
x = df.loc[:,['artist_name']].values
y = df.loc[:, 'Genre'].values

encoder = LabelEncoder()
x = encoder.fit_transform(x)
x = pd.DataFrame(x)

Encoder_y = LabelEncoder()
Y = Encoder_y.fit_transform(y)
Y = pd.DataFrame(Y)

X_train, X_test, y_train, y_test = train_test_split(x, y , test_size = 0.3,
                                                    random_state = 1)

#scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


#KNN Classification

knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

error = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i!=y_test))
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(range(1,30), error, color = 'black',
        marker = 'o', markerfacecolor = 'cyan',
        markersize = 10)
ax.set_xlabel('K Value')
ax.set_ylabel('Mean Error')
ax.set_title('Error Rate K Value')

#another approach

x = df.loc[:, ['Energy','Length','Danceability',
               'beats_per_minute','Acousticness']].values
y = df.loc[:,'Popularity'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output)

#testing the accuracy of Naive bayes
scores = cross_val_score(gnb, X_train, y_train, scoring = 'accuracy',
                        cv = 3).mean()*100
print(scores)

sns.jointplot(x = y_test, y = y_pred, kind = 'kde', color = 'r')


#try another approach

x = df.loc[:, ['Energy','Length','Danceability','beats_per_minute',
               'Acousticness']].values
               
y = df.loc[:, 'Popularity'].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

LinSVC = LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual =True)
LinSVC.fit(X_train, y_train)
y_pred = LinSVC.predict(X_test)
df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output)

scores = cross_val_score(LinSVC, X_train, y_train,
                         scoring = 'accuracy', cv = 3).mean()*100
                         
print(scores)
sns.jointplot(x = y_test, y = y_pred, kind = 'reg', color = 'blue')
