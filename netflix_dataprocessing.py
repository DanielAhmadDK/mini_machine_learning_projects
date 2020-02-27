# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:07:57 2020

@author: s133016
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\netflix_shows\netflix_titles_nov_2019.csv')

df.head()

#for col in df:
#    print(df[col].unique())

#This shows the type of data we have and how many missing values
def data_inv(df):
    print('netflix movies and shows: ',df.shape[0])
    print('dataset variables: ',df.shape[1])
    print('-'*10)
    print('dateset columns: \n')
    print(df.columns)
    print('-'*10)
    print('data-type of each column: \n')
    print(df.dtypes)
    print('-'*10)
    print('missing rows in each column: \n')
    c=df.isnull().sum()
    print(c[c>0])
data_inv(df)

#find how many rows have the same title, country, type and release year
dups=df.duplicated(['title','country','type','release_year'])
df[dups]

#drop these duplicates
df=df.drop_duplicates(['title','country','type','release_year'])

#drop column that does not have any importance
df=df.drop('show_id',axis=1)


#Now deal with missing values
df['cast']=df['cast'].replace(np.nan,'Unknown')
def cast_counter(cast):
    if cast=='Unknown':
        return 0
    else:
        lst=cast.split(', ')
        length=len(lst)
        return length
df['number_of_cast']=df['cast'].apply(cast_counter)
df['cast']=df['cast'].replace('Unknown',np.nan)


df['rating']=df['rating'].fillna(df['rating'].mode()[0])
df['date_added']=df['date_added'].fillna('January 1, {}'.format(str(df['release_year'].mode()[0])))

import re
months={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}
date_lst=[]
for i in df['date_added'].values:
    str1=re.findall('([a-zA-Z]+)\s[0-9]+\,\s[0-9]+',i)
    str2=re.findall('[a-zA-Z]+\s([0-9]+)\,\s[0-9]+',i)
    str3=re.findall('[a-zA-Z]+\s[0-9]+\,\s([0-9]+)',i)
    date='{}-{}-{}'.format(str3[0],months[str1[0]],str2[0])
    date_lst.append(date)
    
    
df['date_added_cleaned']=date_lst
df=df.drop('date_added',axis=1)
df['date_added_cleaned']=df['date_added_cleaned'].astype('datetime64[ns]')



for i in df.index:
    if df.loc[i,'rating']=='UR':
        df.loc[i,'rating']='NR'
        
        
#Visualize the data

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(df['rating'])


#Using one-out-k encoder
from sklearn.preprocessing import OneHotEncoder
from math import sqrt
#find the categorical variables in ratings
df["rating"] = df["rating"].astype('category')

df["rating_cat"] = df["rating"].cat.codes


from sklearn.preprocessing import LabelBinarizer

lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(df["rating"])


ratingLabels = df['rating']
ratingNames = sorted(set(ratingLabels))
ratingDict = dict(zip(ratingNames, range(13)))
rating_Y = np.asarray([ratingDict[value] for value in ratingLabels])

k = 1

df['G'] = [k if value == 0 else 0 for value in rating_Y]  
df['NC-17'] = [k if value == 1 else 0 for value in rating_Y]    
df['NR'] = [k if value == 2 else 0 for value in rating_Y]
df['PG'] = [k if value == 3 else 0 for value in rating_Y]  
df['PG-13'] = [k if value == 4 else 0 for value in rating_Y]  
df['R'] = [k if value == 5 else 0 for value in rating_Y]  
df['TV-14'] = [k if value == 6 else 0 for value in rating_Y]  
df['TV-G'] = [k if value == 7 else 0 for value in rating_Y]  
df['TV-MA'] = [k if value == 8 else 0 for value in rating_Y]  
df['TV-PG'] = [k if value == 9 else 0 for value in rating_Y]  
df['TV-Y'] = [k if value == 10 else 0 for value in rating_Y] 
df['TV-Y7'] = [k if value == 11 else 0 for value in rating_Y]
df['TV-Y7-FV'] = [k if value == 12 else 0 for value in rating_Y]  





Total = [df['G'].sum(), df['NC-17'].sum(), df['NR'].sum(), 
         df['PG'].sum(), df['PG-13'].sum(), df['R'].sum(),
         df['TV-14'].sum(), df['TV-G'].sum(), df['TV-MA'].sum(),
         df['TV-PG'].sum(), df['TV-Y'].sum(), df['TV-Y7'].sum(),
         df['TV-Y7-FV'].sum()]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.pie(Total, labels = ratingNames, autopct='%1.1f%%')


import seaborn as sns
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

sns.countplot(x = 'rating', hue = 'type', data = df)


#check the country with the most movies/series

df['country'].value_counts().sort_values(ascending = False)


top_productive_countries=df[(df['country']=='United States')|(df['country']=='India')|(df['country']=='United Kingdom')|(df['country']=='Japan')|
                             (df['country']=='Canada')|(df['country']=='Spain')]
plt.figure(figsize=(10,8))
sns.countplot(x='country',hue='type',data=top_productive_countries)
plt.title('comparing between the types that the top countries produce')
plt.show()




#This shows the percentage of ratings for each of the most popular countries

for i in top_productive_countries['country'].unique():
    print(i)
    print(top_productive_countries[top_productive_countries['country']==i]['rating'].value_counts(normalize=True)*100)
    print('-'*10)
    
    
#show how many movies vs series

df['year_added']=df['date_added_cleaned'].dt.year
df['type'].value_counts(normalize=True)*100


df.groupby('year_added')['type'].value_counts(normalize=True)*100

dups=df.duplicated(['title'])
df[dups]['title']


for i in df[dups]['title'].values:
    print(df[df['title']==i][['title','type','release_year','country']])
    print('-'*40)
    
    
    
    
plt.figure(figsize=(10,8))
df['year_added'].value_counts().plot.bar()
plt.title('distribution of year-added')
plt.ylabel('relative frequency')
plt.xlabel('year_added')
plt.show()




counts=0
for i,j in zip(df['release_year'].values,df['year_added'].values):
    if i!=j:
        counts+=1
print('number of contents that its release year differ from the year added to netflix are ',str(counts))