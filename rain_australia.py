# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:17:00 2020

@author: s133016
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\Rain_Australia\weatherAUS.csv')


df_1000 = df.sample(n = 1000)
#start pre-processing
def data_inv(df):
    print('Australia rain dataset: ',df.shape[0])
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


#TASK:Predict whether or not it will rain tomorrow
#by training a binary classification model on target RainTomorrow
# y = RainTomorrow

#Note: Exclude Risk-MM

df.drop('RISK_MM', axis = 1)
df_1000.drop('RISK_MM', axis = 1, inplace = True)

df.describe(include = 'all')
describe = df.describe()

#check if target has any missing values

df['RainTomorrow'].isnull().sum()# there are no missing values
if df['RainTomorrow'].isnull().sum() == 0:
    print('Target has no missing values')
else:
    print('Missing values are:', df['RainTomorrow'].isnull().sum())
    
#view unique values of target
    
df['RainTomorrow'].unique() #this shows the unique values
df['RainTomorrow'].nunique() #this shows the actual number


#find the frequency of no and yes
df['RainTomorrow'].value_counts() # 110316 No ;  31877 Yes

#view the distribution
df['RainTomorrow'].value_counts()/len(df)*100 #77.6% No and 22.5% Yes


#view the distribution
fig, ax = plt.subplots(figsize = (6,8))
ax = sns.countplot(x = 'RainTomorrow', data = df, palette = 'Set1' )


fig, ax = plt.subplots(figsize = (6,8))
ax = sns.countplot(y = 'RainTomorrow', data = df, palette = 'Set1' )



#Explore categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :' ,  categorical)


#find missing categorical variables
df[categorical].isnull().sum()

#print categorical variables containing missing values
cat1 = [var for var in categorical if df[var].isnull().sum()!=0] #prints the cat variables that are not 0
print(df[cat1].isnull().sum())

#view the frequency of categorical variables
for var in categorical:
    print(df[var].value_counts())


for var in categorical:
    print(df[var].value_counts()/len(df)*100)
    
for var in categorical:
    print(var, 'contains', len(df[var].unique()), 'labels')
    
#date variable needs to be preprocessed because there is around 3400 unique lables
#this is a high cardinality
    
df['Date'].dtypes

#parse the dates
#create month and day column
df_1000['Date'] = pd.to_datetime(df_1000['Date'])   
df_1000['Year'] = df_1000['Date'].dt.year
df_1000['Month'] = df_1000['Date'].dt.month   
df_1000['Day'] = df_1000['Date'].dt.day

#now we can drop the Date column
df_1000.drop('Date', axis = 1, inplace =True)


#Now we can explore categorical variables one by oone
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are:' , categorical)

#check missing values again for categorical data
df[categorical].isnull().sum()

#WindGustDir, WindDir9am, WindDir3pm, RainToday contain missing values
#we need to explore one by one

print('Location contains', len(df.Location.unique()), 'labels')
df.Location.unique()

df.Location.value_counts()
df.Location.value_counts()/len(df)*100
pd.get_dummies(df.Location, drop_first = True)
#do one hot encoding of location variable
#Location_OneHot = df['Location']
#Location_OneHot = np.asarray(Location_OneHot)
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder


#label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(Location_OneHot)


#onehot_encoder = OneHotEncoder(sparse =False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


#Location_OH_changed = pd.DataFrame(onehot_encoded)

#df_new = pd.concat([Location_OH_changed, df])
#explore WindGustDir variable
print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')

df['WindGustDir'].unique()
df['WindGustDir'].value_counts()
pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True)

pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).sum(axis = 0)


#explore WindDir9am variable
print('WindDiram contains', len(df.WindDir9am.unique()), 'labels')
df['WindDir9am'].value_counts()

pd.get_dummies(df.WindDir9am, drop_first = True, dummy_na = True)


#the same procedure would be applied for
#the remaining categorical variables


#Now we explore the numerical variables
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are:', numerical)

df[numerical].isnull().sum()

#view summary statistics in numerical variables
summary_numerical = round(df[numerical].describe(), 2)

#visualize the outliers
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df.boxplot(column = 'Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2,2,2)
fig = df.boxplot(column = 'Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig = df.boxplot(column = 'WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2,2,4)
fig = df.boxplot(column = 'WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


#check the distribution of variables
plt.figure(figsize = (15,10))

plt.subplot(2,2,1)
fig = df.Rainfall.hist(bins = 10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,2)
fig = df.Evaporation.hist(bins = 10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,3)
fig = df.WindSpeed9am.hist(bins = 10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,4)
fig = df.WindSpeed3pm.hist(bins = 10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')

#all variables are skewed and there is no normal
#distribution. Now we find outliers starting with
#Rainfall varaiable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_bound = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_bound = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(
        lowerboundary = Lower_bound, upperboundary = Upper_bound))

#so outliers for Rainfall are > 3.2

#we look at Evaporation variable
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_bound = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_bound = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(
        lowerboundary = Lower_bound, upperboundary = Upper_bound))

#outliers for Evaporation are > 21.8
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_bound = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_bound = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(
        lowerboundary = Lower_bound, upperboundary = Upper_bound))

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_bound = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_bound = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(
        lowerboundary = Lower_bound, upperboundary = Upper_bound))


#-------Multivariate analysis----
#we will use a heat map and a pair plot
#to discover the relationship

correlation = df_1000.corr()

plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.1f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
ax.set_ylim(len(correlation)-1, -1)           
plt.show()

#now make a pairs plot
num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 
           'Temp3pm', 'WindGustSpeed',
           'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']

sns.pairplot(df_1000[num_var], kind = 'scatter',
             diag_kind = 'hist', palette = 'Rainbow', plot_kws={'s':10})
plt.show()

#-------Machine learning starts now-------
#Remove target from dataframe

X =df_1000.drop(['RainTomorrow'], axis = 1)
y = df_1000['RainTomorrow']

#now we split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train.shape, X_test.shape

#Feature engineering
#check data types in X_train

X_train.dtypes

categorical = [col for col in X_train.columns if X_train[col].dtypes=='O']
categorical

numerical = [col for col in X_train.columns if X_train[col].dtypes!= 'O']
numerical

#Check for missing values
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()

#print percentage of missing values
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean()*100, 4))
        
#deal with missing values
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace =True)
        
X_train[numerical].isnull().sum()#now there are no missing values
X_test[numerical].isnull().sum()# no missing values

#print % of missing values in categorical training set
X_train[categorical].isnull().mean()
for col in categorical:
    if X_train[col].mean()>0:
        print(col, (X_train[col].isnull().mean()))
        
#impute missing categorical variables with most frequent

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
    
#check missing values in categorical variables
X_train[categorical].isnull().sum()
X_test[categorical].isnull().sum()
X_train.isnull().sum()
X_test.isnull().sum()
#no missing values, so we can proceed

#Now we deal with outliers in numerical variables
#we will use top-coding approach to cap max values
#and remove outliers from the above variables
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
    
X_train.Rainfall.max(), X_test.Rainfall.max()
X_train.Evaporation.max(), X_test.Evaporation.max()
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()

#now the max values for the selected variables are
#the ones found.
X_train[numerical].describe()

#Encode categorical variables
categorical

import category_encoders as ce

encoder = ce.BinaryEncoder(cols = ['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

#we now added to columns 0 for no rain
#1 for it is raining

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                    pd.get_dummies(X_train.Location),
                    pd.get_dummies(X_train.WindGustDir),
                    pd.get_dummies(X_train.WindDir9am),
                    pd.get_dummies(X_train.WindDir3pm)], axis = 1)

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                    pd.get_dummies(X_test.Location),
                    pd.get_dummies(X_test.WindGustDir),
                    pd.get_dummies(X_test.WindDir9am),
                    pd.get_dummies(X_test.WindDir3pm)], axis = 1)
    
#we need to now convert the variables
#to the same scale
    
cols = X_train.columns
    
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns = [cols])
X_test = pd.DataFrame(X_test, columns = [cols])

#-----Model training

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'liblinear', random_state = 0)
logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)
#y_pred_test

#probability of getting output as 0 - no rain

#logreg.predict_proba(X_test)[:,0]*100
#logreg.predict_proba(X_test)[:,1]*100

#check the accuracy

from sklearn.metrics import accuracy_score

#print('The model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))


y_pred_train = logreg.predict(X_train)

y_est_test = np.asarray(logreg.predict(X_test))
y_est_train = np.asarray(logreg.predict(X_train))
misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))

print('Training accuracy:{0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))


#check for overfitting and underfitting
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))


#Now we start to adjust logreg parameters C

logreg100 = LogisticRegression(C=100, solver='liblinear',random_state=0)
logreg100.fit(X_train, y_train)

print('Training set score:{:.4f}'.format(logreg100.score(X_train, y_train)))
print('Test set score:{:.4f}'.format(logreg100.score(X_test, y_test)))


#compare model with base 
y_test.value_counts()
#most frequent class is 22067 which is No
#so we can calculate null accuracy by dividing
#22067 by total number of occurences

null_accuracy = (22067/(22067+6372))
print('Null accuracy score: {0:0.4f}'.format(null_accuracy))


#we now use a confusion matric
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP)=', cm[0,0])
print('\nTrue Negatives(TN)=', cm[1,1])
print('\nFalse Positives(FP)=', cm[0,1])
print('\nFalse Negatives(FN)=', cm[1,0])

cm_matrix = pd.DataFrame(data = cm, columns = ['Actual Positive: 1','Actual Negative: 0'],
                         index = ['Predict Positive: 1', 'Predict Negative: 0'])
fig, ax = plt.subplots()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu', ax = ax)



#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))

TP = cm[0,0]
TN =cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP+TN) / float(TP + TN + FP + FN)
print('Classification accuracy: {0:0.4f}'.format(classification_accuracy))

#classifcation error

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error:{0:0.4f}'.format(classification_error))


#print the precision score

precision = TP / float(TP + FP)
print('Precision: {0:0.4f}'.format(precision))

#print the recall score

recall = TP / float(TP + FN)
print('Recall or Sensitivity:{0:0.4f}'.format(recall))

#true positive rate is the same as recall
#false positive rate
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP +TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

#---Adjusting the threshold level
y_pred_prob = logreg.predict_proba(X_test)[0:10]

#there are two columns, with each row summing to 1
y_pred_prob_df = pd.DataFrame(data = y_pred_prob, columns = ['Prob of - No rain tomorrow (0)',
                                                             'Prob of - Rain tomorrow (1)'])
    
    
logreg.predict_proba(X_test)[0:10,1]
y_pred1 = logreg.predict_proba(X_test)[:,1]

#make a histogram
plt.rcParams['font.size'] = 12
plt.hist(y_pred1, bins = 10)
plt.title('Histogram of predicted probabilities of rain')
plt.xlim(0,1)
plt.xlabel('Predicted probabilites of rain')
plt.ylabel('Frequency')


#Testing different thresholds

from sklearn.preprocessing import binarize
for i in range(1,5):
    cm1 = 0
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    y_pred1 = y_pred1.reshape(-1,1)
    y_pred2 = binarize(y_pred1, i/10)
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    cm1 = confusion_matrix(y_test, y_pred2)
    print('With', i/10, 'threshold the Confusion Matrix is', '\n\n', cm1, '\n\n',
                      'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
    
    
#ROC and AUC curves
  
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize = (6,4))
plt.plot(fpr, tpr, linewidth = 2)
plt.plot([0,1], [0,1], 'k--')
plt.rcParams['font.size'] = 12
plt.title('ROC curve for RainTomorrow classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.show()

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))



#Model improvement methods
#We discuss feature selection and cross validation

from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=logreg, step=1, cv=5, scoring='accuracy')

rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features: %d' % rfecv.n_features_)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.feature_selection import SelectKBest # Univariate Feature Selection
from sklearn.feature_selection import chi2 # To apply Univariate Feature Selection
from sklearn.feature_selection import RFE # Recursive Feature Selection
from sklearn.decomposition import PCA # To apply PCA
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#----------Now I will try feature selection methods

UnivariateFeatureSelection = SelectKBest(chi2, k=5).fit(X_train, y_train)

dic = {key:value for (key, value) in zip(UnivariateFeatureSelection.scores_, X_train.columns)}
sorted(dic.items(), reverse =True)


#extracting best K-values
X_train_k_best = UnivariateFeatureSelection.transform(X_train)
X_test_k_best = UnivariateFeatureSelection.transform(X_test)


#Random forest classifier
RandomForest_K_best = RandomForestClassifier()
RandForest_K_best = RandomForest_K_best.fit(X_train_k_best, y_train)

y_pred = RandForest_K_best.predict(X_test_k_best)
score = accuracy_score(y_test, y_pred)
print('Score is:', score)

RandForest_RFE = RandomForestClassifier()
rfe = RFE(estimator = RandForest_RFE, n_features_to_select = 5, step = 1)
rfe = rfe.fit(X_train, y_train)

print('Best features chosen by RFE: \n')
for i in X_train.columns[rfe.support_]:
    print(i)
    
RandForest_RFECV = RandomForestClassifier()
rfecv = RFECV(estimator = RandForest_RFECV, step = 1, cv = 3, scoring = 'accuracy')
rfecv = rfecv.fit(X_train, y_train)
print('Best number of features:', rfecv.n_features_)
print('Features :\n')
for i in X_train.columns[rfecv.support_]:
    print(i)
    
plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("Score of Selected Features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(1,1,1)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



#------------MORE FEATURE SELECTION--------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_features = df.drop(['RainTomorrow'], axis = 1)
y_target = df['RainTomorrow']


categorical = [col for col in X_features.columns if X_features[col].dtypes=='O']
categorical

numerical = [col for col in X_features.columns if X_features[col].dtypes!= 'O']
numerical



#we now added to columns 0 for no rain
#1 for it is raining

for df1 in [X_features]:
    for col in numerical:
        col_median = X_features[col].median()
        df1[col].fillna(col_median, inplace =True)

for df2 in [X_features]:
    df2['WindGustDir'].fillna(X_features['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_features['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_features['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_features['RainToday'].mode()[0], inplace=True)


import category_encoders as ce

encoder = ce.BinaryEncoder(cols = ['RainToday'])
X_features = encoder.fit_transform(X_features)

#we now added to columns 0 for no rain
#1 for it is raining

X_features = pd.concat([X_features[numerical], X_features[['RainToday_0', 'RainToday_1']],
                    pd.get_dummies(X_features.Location),
                    pd.get_dummies(X_features.WindGustDir),
                    pd.get_dummies(X_features.WindDir9am),
                    pd.get_dummies(X_features.WindDir3pm)], axis = 1)
#apply SelectKBest class to extract top 10 best features

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X_features, y_target)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_,
                                index = X_features.columns)
feat_importances.nlargest(10).plot(kind = 'barh')
plt.show()
