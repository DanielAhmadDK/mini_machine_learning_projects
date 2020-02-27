# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:23:48 2020

@author: s133016
"""

import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

#----------------Reading the data--------------
df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\heart_disease_two\heart.csv')

def detailed_analysis(df, pred=None):
  obs = df.shape[0]
  types = df.dtypes
  counts = df.apply(lambda x: x.count())
  uniques = df.apply(lambda x: [x.unique()])
  nulls = df.apply(lambda x: x.isnull().sum())
  distincts = df.apply(lambda x: x.unique().shape[0])
  missing_ratio = (df.isnull().sum() / obs) * 100
  skewness = df.skew()
  kurtosis = df.kurt()
  print('Data shape:', df.shape)

  if pred is None:
    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)
  else:
    corr = df.corr()[pred]
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis, corr], axis=1, sort=False)
    corr_col = 'corr ' + pred
    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis', corr_col]

  details.columns = cols
  dtypes = details.types.value_counts()
  print('____________________________\nData types:\n', dtypes)
  print('____________________________')
  return details

details = detailed_analysis(df, 'target')
display(details.sort_values(by = 'corr target', ascending = False))


#fbs, chol are not significant and could be considered to remove
df_new = df.drop(['chol','fbs'], axis = 1)
import seaborn as sns

correlation = df.corr()
ax = sns.heatmap(correlation,fmt='.2f', annot = True,
            vmin = -1, vmax = 1, center = 1)
ax.set_ylim(len(correlation)-1, -1)    

df_new['gender'] = ['Male' if value == 0 else 'Female' for value in df_new['sex']]
df_new['Heart_Disease'] = ['Positive' if value == 1 else 'Negative' for value
      in df_new['target']]

fig = plt.figure(figsize = (10,10))
ax = sns.boxplot('gender', 'age', hue = 'target', data = df_new)

fig = plt.figure(figsize = (10,10))
ax = sns.scatterplot(x = 'age', y = 'trestbps', hue = 'Heart_Disease', data = df_new)

df_new.groupby('Heart_Disease').size()
df_new.groupby('gender').size()
df_new.groupby(['Heart_Disease', 'gender']).size()

#---Now the setting up the pipeline begins---------

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

num_col = [col for col in df_new.columns if df_new[col].dtypes != 'O']
cat_col = [col for col in df_new.columns if df_new[col].dtypes == 'O']

X = df_new.drop(['target'], axis = 1)
y = df_new['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


#----Transforming the scale of the data-----

cols = X_train.columns   
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = [cols])
X_test = pd.DataFrame(X_test, columns = [cols])


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_mdl = LogisticRegression()
log_mdl.fit(X_train, y_train)
y_pred = log_mdl.predict(X_test)

print('Training accuracy:{0:0.4f}'.format(accuracy_score(y_test, y_pred)))

#--------------Apply different models to obtain the model-----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.feature_selection import SelectKBest # Univariate Feature Selection
from sklearn.feature_selection import chi2 # To apply Univariate Feature Selection
from sklearn.feature_selection import RFE # Recursive Feature Selection



seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Random', RandomForestClassifier()))

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

##---------------------Feature selection method------------

from sklearn.feature_selection import RFECV
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

#------------------Way to see the optimal k value for KNN---------
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP)=', cm[0,0])
print('\nTrue Negatives(TN)=', cm[1,1])
print('\nFalse Positives(FP)=', cm[0,1])
print('\nFalse Negatives(FN)=', cm[1,0])

cm_matrix = pd.DataFrame(data = cm, columns = ['Actual Positive: 1','Actual Negative: 0'],
                         index = ['Predict Positive: 1', 'Predict Negative: 0'])
fig, ax = plt.subplots()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu', ax = ax)

TP = cm[0,0]
TN =cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP+TN) / float(TP + TN + FP + FN)
print('Classification accuracy: {0:0.4f}'.format(classification_accuracy))

#classifcation error

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error:{0:0.4f}'.format(classification_error))


#-------------------Cross validation with loop---------
N,M = X.shape
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.StratifiedKFold(n_splits=K)

# Initialize variables
Error_logreg = np.empty((K,1))
Error_dectree = np.empty((K,1))
n_tested=0

k=0
for train_index, test_index in CV.split(X,y):
    print('CV-fold {0} of {1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit and evaluate Logistic Regression classifier
    model = LogisticRegression(C=N)
    model = model.fit(X_train, y_train)
    y_logreg = model.predict(X_test)
    Error_logreg[k] = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
    
    # Fit and evaluate Decision Tree classifier
    model2 = DecisionTreeClassifier()
    model2 = model2.fit(X_train, y_train)
    y_dectree = model2.predict(X_test)
    Error_dectree[k] = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)

    k+=1

# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
z = (Error_logreg-Error_dectree)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

from scipy import stats

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
    
#-------Super learner method-----
    
import pandas as pd
from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray

df = pd.read_csv(r'C:\Users\danie\Desktop\ML_project\heart_disease_two\heart.csv')
df_new = df.drop(['chol','fbs'], axis = 1)

X = df_new.drop(['target'], axis = 1)
y = df_new['target']

X = asarray(X)
y = asarray(y)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import mean_squared_error
#create a list of base-models

def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(SVR(gamma = 'scale'))
    models.append(DecisionTreeRegressor())
    models.append(KNeighborsRegressor())
    models.append(AdaBoostRegressor())
    models.append(BaggingRegressor(n_estimators = 10))
    models.append(RandomForestRegressor(n_estimators = 10))
    models.append(ExtraTreesRegressor(n_estimators = 10))
    return models

#collect out of fold predictions form k-fold cross validation

def get_out_of_fold_predictions(X_train, y_train, models):
    meta_X, meta_y = list(), list()
    #define split of data
    kfold = KFold(n_splits = 10, shuffle = True)
    #enumerate splits
    for train_ix, test_ix in kfold.split(X):
        fold_ypred = list()
        #get data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        meta_y.extend(y_test)
        #fit and make predictions with each sub-model
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            #store columns
            fold_ypred.append(y_pred.reshape(len(y_pred),1))
        #store fold y_pred as columns
        meta_X.append(hstack(fold_ypred))
    return vstack(meta_X), asarray(meta_y)

#fit all base models on the training dataset
def fit_base_models(X, y, models):
    for model in models:
        model.fit(X,y)
        
#fit a meta modle
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X,y)
    return model

#evaluate a list of models on a dataset
def evaluate_models(X, y , models):
    for model in models:
        y_pred = model.predict(X)
        mse = mean_squared_error(y_test, y_pred)
        print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))
        
#make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for models in models:
        y_pred = models.predict(X)
        meta_X.append(y_pred.reshape(len(y_pred),1))
    meta_X = hstack(meta_X)
    #predict
    return meta_model.predict(meta_X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
models = get_models()
meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
print('meta', meta_X.shape, meta_y.shape)

fit_base_models(X_train, y_train, models)

meta_model = fit_meta_model(meta_X, meta_y)

evaluate_models(X_test, y_test, models)

y_pred = super_learner_predictions(X_test, models, meta_model)

print('Super learner: RMSE %.3f' % (sqrt(mean_squared_error(y_test, y_pred))))
