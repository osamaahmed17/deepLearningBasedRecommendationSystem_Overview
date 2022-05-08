#!/usr/bin/env python
# coding: utf-8

# **Note:** *This the Master's thesis version of the code as the main was given to Ford's staff and can't be shared due to NDA (Non Disclosure Agreement)*

# In[ ]:


import pandas as pd
import numpy as np
import time
import keras
from utils import *
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# In[ ]:


dataset = pd.read_csv("u.data",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
print(dataset.shape)

dataset.head()


# # Non Matrix Factorization

# In[ ]:


##############Non Matrix Factorization##############
R_shape = getShape('u.data') 
X, y, R = loadData('u.data', R_shape) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_users = len(dataset['user_id'].unique())
n_items = len(dataset['item_id'].unique())
R_shape = (n_users, n_items)
R = convertToDense(X, y, R_shape)
parametersNMF = {
                    'n_components' : 20,     # number of latent factors
                    'init' : 'random', 
                    'random_state' : 0, 
                    'alpha' : 0.01,          # regularization term
                    'l1_ratio' : 0,          # set regularization = L2 
                    'max_iter' : 10
                }

estimator = NMF(**parametersNMF)


err = 0
n_iter = 0.
no_splits = 5

fold = KFold(no_splits,shuffle=False)

for train_index, test_index in fold.split(X): 
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    R_train = convertToDense(X_train, y_train, R_shape)
    R_test = convertToDense(X_test, y_test, R_shape)

    t0 = time.time()
    estimator.fit(R_train)  
    Theta = estimator.transform(R_train)       
    M = estimator.components_.T                
    n_iter += estimator.n_iter_ 

    R_pred = M.dot(Theta.T)
    R_pred = R_pred.T      
    
    R_pred[R_pred > 5] = 5.                  
    R_pred[R_pred < 1] = 1.           

    err += getRMSE(R_pred, R_test)
    
print ("Non Matrix Factorization RMSE Error : ", err / no_splits)


# # Neural Rating Regressing

# In[ ]:


############## Neural Rating Regressing ##############
estimator = KerasRegressor(build_fn=neuralRatingRegression, nb_epoch=100, batch_size=100, verbose=False)
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_train)
kfold = KFold(n_splits=no_splits, random_state=1,shuffle=True )
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
mse_krr = mean_squared_error(y_train, y_pred)
err = np.sqrt(mse_krr)
    
print("Neural Rating Regression RMSE Error : ",err)


# # AutoRec

# In[ ]:


############## AutoRec ##############
estimator = autoRec()
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_train)
kfold = KFold(n_splits=no_splits, random_state=1,shuffle=True )
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
mse_krr = mean_squared_error(y_train, y_pred)
err = np.sqrt(mse_krr)
    
print("AutoRec RMSE Error : ",err)


# # Matrix Factorization

# In[ ]:


############## Matrix Factorization ##############
dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
train, test = train_test_split(dataset, test_size=0.33)
print(train.shape, test.shape)

X_train, X_test, y_train, y_test = train_test_split(dataset[['user_id', 'item_id']], dataset['rating'], test_size=0.33)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
n_latent_factors = 20
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input))
prod = keras.layers.dot([movie_vec, user_vec], axes=1,name='DotProduct')
model = keras.Model([user_input, movie_input], prod)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

history = model.fit(list(np.transpose(X_train.values)), y_train, epochs=20, verbose=0)
pred = model.predict(list(np.transpose(X_train.values)))
mse_krr = mean_squared_error(train.rating, pred)
err = np.sqrt(mse_krr)
print("MF RMSE Error : ",err)

