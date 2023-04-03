import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# print correlation matrix
print(dataframe.corr()) 

# split into input (X) and output (y) variables
X, y = dataset[:, 1:-1], dataset[:, -1]
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#--- main algorithm

print("\np\tseed\terror\tdtime\tsize")
stats = {}

for p in np.arange(0.01, 1.0001, 0.01):
    for seed in range(203,213):

            # use seed to check volatility between identical runs
            np.random.seed(seed)     

            # keep proportion p of the original training set 
            X_train_small = X_train  
            y_train_small = y_train
            max_size = len(X_train_small)
            while len(X_train_small) >= p * max_size:
                index = np.random.randint(len(X_train_small))
                X_train_small = np.delete(X_train_small, (index), axis=0)
                y_train_small = np.delete(y_train_small, (index), axis=0)

            # run model on small training set, save history
            model = LinearRegression(fit_intercept = True)
            time1 = time.time()         
            model.fit(X_train_small, y_train_small)
            time2 = time.time()
            dtime = time2 - time1
            train_size = len(X_train_small)/len(X_train)
            coeff = model.coef_

            # evaluate on training set
            y_pred_train = model.predict(X_train_small)
            error_train = metrics.mean_absolute_error(y_train_small, y_pred_train)

            # evaluate on test set
            y_pred_test = model.predict(X_test)
            error_test = metrics.mean_absolute_error(y_test, y_pred_test)
            stats[(p,seed)] = (error_test, error_train, dtime, model.coef_)
            print("%5.4f\t%4d\t%6.4f\t%8.6f" % (p, seed, error_test, dtime))

#--- improve (recalibrate) predictions obtained via last regression

def create_pred_correction_table(observed, predicted):
    delta = {}
    count = {}
    for k in range(len(predicted)):
        pred_value = int(0.5 + predicted[k])
        if pred_value in delta:
            delta[pred_value] += pred_value - observed[k]
            count[pred_value] += 1
        else:
            delta[pred_value] = pred_value - observed[k]
            count[pred_value] = 1
    for pred_value in delta:
        delta[pred_value] /= count[pred_value]
    return(delta)

def adjusted_prediction(pred, delta):
    pred_int = int(0.5 + pred)
    if pred_int in delta: 
        pred_adjusted = pred + delta[pred_int]
    else:
        pred_adjusted = pred
    return(pred_adjusted)

delta = create_pred_correction_table(y_pred_train, y_train_small)

# recalibrate predictions
y_pred_test_adj = []
for k in range(len(y_pred_test)):
    pred = y_pred_test[k]
    adj_pred = adjusted_prediction(pred, delta)
    y_pred_test_adj.append(adj_pred)

#--- plot last regression obtained on full training set (p = 1), and correls

plt.xlim(0,25)
plt.ylim(-5,35)
plt.plot([-5, 35],[-5, 35], c='blue', linewidth = 0.5)
plt.scatter(y_test,y_pred_test_adj, s=28, c='r', edgecolors='r', \
    linewidth=0, alpha = 0.2) 
plt.show()
c1 = np.corrcoef(y_train_small,y_pred_train)
c2 = np.corrcoef(y_test,y_pred_test)
c3 = np.corrcoef(y_test,y_pred_test_adj)
print("\nCorrels:\non training set: %5.4f\non test set: %5.4f\non test set adj: %5.4f" 
     % (c1[0,1],c2[0,1],c3[0,1]) )        

#--- compute aggregated statistics (one set for each p, computed across seeds)

avg_error_test = {}     # avg X_test error given p
min_error_test = {}     # min X_test error given p
max_error_test = {}     # max X_test error given p
min_error_train = {}    # min X_train_small error given p
min_error3 = {}    # X_test error for best_seed2 (minimizing X_train_small error given p)
best_seed ={}      # seed that minimizes X_test error given p
best_seed2 = {}    # seed that minimizes X_train_small error given p
count = {}         # number of seeds used for specific p

for key in stats:
    p    = key[0]
    seed = key[1]
    statistics = stats[key]
    error_test = statistics[0]    # computed on test (validation) set
    error_train = statistics[1]   # computed on small training set
    arr_coeff = statistics[3]
    if p in avg_error_test:
       avg_error_test[p] += error_test
       if error_test < min_error_test[p]:
           min_error_test[p] = error_test
           best_seed[p] = seed
       if error_test > max_error_test[p]:
           max_error_test[p] = error_test
       if error_train < min_error_train[p]:
           min_error_train[p] = error_train
           best_seed2[p] = seed
           min_error3[p] = error_test
       count[p] +=1
    else:
       avg_error_test[p]  = error_test
       min_error_test[p]  = error_test
       max_error_test[p]  = error_test
       min_error_train[p] = error_train
       min_error3[p] = error_test
       best_seed[p]  = seed
       best_seed2[p] = seed
       count[p] = 1

# print summary stats
print("\nAggregated stats\np\tcount\tavg err test\tmin err test \
           \tmax err test\tseed1\tmin err train\tmin err3\tseed2")
for p in count:
    avg_error_test[p] /= count[p]
    print("%5.4f\t%3d\t%7.5f\t%7.5f\t%7.5f\t%4d\t%7.5f\t%7.5f\t%4d"
        % (p, count[p], avg_error_test[p], min_error_test[p], max_error_test[p], 
           best_seed[p], min_error_train[p], min_error3[p], best_seed2[p]))

# for each p, print min_error[p] and the corresponding regression coefficients 
print("\nRegression coefficients, best fit given p")
for p in count: 
    seed = best_seed[p]
    key = (p, seed)
    statistics = stats[(p, seed)]
    error_test = statistics[0] 
    print("%5.4f\t%7.5f\t" % (p, error_test), end=" ")
    regr_coeff = statistics[3]   
    for var in range(len(regr_coeff)):
        print(" %6.2f" % (regr_coeff[var]), end=" ")
    print(" ")
