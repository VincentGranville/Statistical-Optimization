import time
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import random as python_random
from tensorflow import random

# initialize plot parameters for nicer plots
mpl.rcParams['axes.linewidth'] = 0.5
fig = plt.figure() 
axes = plt.axes()
axes.tick_params(axis='both', which='major', labelsize=8)
axes.tick_params(axis='both', which='minor', labelsize=8)

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'
dataframe = read_csv(url, header=None)
dataset = dataframe.values
X_train, y_train = dataset[:, 1:-1], dataset[:, -1]

# split into input (X) and output (y) variables
X, y = dataset[:, 1:-1], dataset[:, -1]
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


header1 = "p\tn_epochs\tseed\terror\tfinal_mae\tdtime\tsize"
print(header1)
OUT = open("history.txt","w")

for p in np.arange(0.10, 1.01, 0.10):
    for n_epochs in range(150, 500, 150):
        for seed in range(103,108):

            # use seed to check volatility between identical runs
            np.random.seed(seed)     # for numpy
            random.set_seed(seed)    # for tensorflow/keras
            python_random.seed(seed) # for python

            # define the keras model
            model = Sequential()
            model.add(Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
            model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
            model.add(Dense(1, activation='linear'))

            # compile the keras model
            model.compile(loss='mse', optimizer='adam', metrics=[metrics.mean_squared_error,
                  metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])

            # keep proportion p of the original training set 
            X_train_small = X_train  
            y_train_small = y_train
            max_size = len(X_train_small)
            while len(X_train_small) >= p * max_size:
                index = np.random.randint(len(X_train_small))
                X_train_small = np.delete(X_train_small, (index), axis=0)
                y_train_small = np.delete(y_train_small, (index), axis=0)

            # run model on small training set, save history
            time1 = time.time()                   
            hist = model.fit(X_train_small, y_train_small, epochs=n_epochs, \
               batch_size=32, verbose=0)
            time2 = time.time()
            dtime = time2 - time1
            mape = hist.history['mean_absolute_percentage_error'] 
            mse  = hist.history['mean_squared_error']
            mae  = hist.history['mean_absolute_error']
            for epoch in range(n_epochs): 
                OUT.write("%4.3f\t%4d\t%4d\t%7.4f\t%7.4f\t%7.4f\t\n" 
                      %(p, seed, epoch, mse[epoch], mae[epoch], mape[epoch]))
            error_train = mae[n_epochs-1]

            # evaluate on test set
            y_pred_test = model.predict(X_test, verbose = 0)
            error_test = mean_absolute_error(y_test, y_pred_test)

            print("%4.3f\t%4d\t%4d\t%6.4f\t%6.4f\t%5.2f\t%4d" % 
                (p, n_epochs, seed, error_test, error_train, dtime, len(X_train_small)))

OUT.close()

#--- plot last predictions obtained, along with associated correls

plt.xlim(0,25)
plt.ylim(-5,35)
plt.plot([-5, 35],[-5, 35], c='blue', linewidth = 0.5)
plt.scatter(y_test,y_pred_test, s=28, c='r', edgecolors='r', linewidth=0, alpha = 0.2) 
plt.show()
y_pred_test = np.transpose(y_pred_test)
c = np.corrcoef(y_test,y_pred_test)
print("\nCorrels:\non test set: %5.4f" % (c[0,1]) )        
