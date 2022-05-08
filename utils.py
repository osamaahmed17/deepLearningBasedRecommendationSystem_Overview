from scipy import sparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def convertToDense(X, y, shape):  # from R=(X,y), in sparse format 
    row  = X[:,0]
    col  = X[:,1]
    data = y
    matrix_sparse = sparse.csr_matrix((data,(row,col)), shape=(shape[0]+1,shape[1]+1))  # sparse matrix in compressed format (CSR)
    R = matrix_sparse.todense()   # convert sparse matrix to dense matrix, same as: matrix_sparse.A
    R = R[1:,1:]                  # removing the "Python starts at 0" offset
    R = np.asarray(R)             # convert matrix object to ndarray object
    return R


def getShape(filename):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(filename, sep='\t', names=names)
    users = len(df['user_id'].unique())
    items = len(df['item_id'].unique())
    return (users, items)

def loadData(filename, R_shape):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(filename, sep='\t', names=names)   
    X = df[['user_id', 'item_id']].values
    y = df['rating'].values   
    return X, y, convertToDense(X, y, R_shape)

def getRMSE(pred, actual):
    pred = pred[actual.nonzero()].flatten()     # Ignore nonzero terms
    actual = actual[actual.nonzero()].flatten() # Ignore nonzero terms
    return np.sqrt(mean_squared_error(pred, actual))


def neuralRatingRegression():
  # create model
  model = Sequential()
  model.add(Dense(500, input_dim=2, activation= "relu"))
  model.add(Dense(100, activation= "relu"))
  model.add(Dense(50, activation= "relu"))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_squared_error"])
  return model
 
def autoRec():   
    # Encoder structure
    n_encoder1 = 500
    n_encoder2 = 300
    n_latent = 2
    # Decoder structure
    n_decoder2 = 300
    n_decoder1 = 500
    model = MLPRegressor(hidden_layer_sizes = (n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1), 
                   activation = 'relu', 
                   solver = 'adam', 
                   learning_rate_init = 0.0001, 
                   max_iter = 10, 
                   tol = 0.0000001, 
                   verbose = False)
    return model
