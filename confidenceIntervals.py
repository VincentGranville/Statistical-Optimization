import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm

M = 20      # number of subsamples
p = 0.025   # confidence level for CI (0 < p < 0.50)
data = []
data_raw = []
CI_nobs = []
CI_lower = []
CI_upper = []
CI_mean = [] 
CI_width = []

#--- create dataset with 2 features; theoretical correl = 0.05 

b1 = -1 + np.sqrt(5) / 2
b2 = 2 / np.sqrt(5)
seed = 67
np.random.seed(seed)
[x1, x2] = [0, 0]

mode = 'Beatty'   # options: 'Beatty' or 'Random'
if mode == 'Beatty':
    N = 5000   # size of full dataset
else:
    N = 50000  # size of full dataset

for k in range(N): 
    if mode == 'Beatty':
        x1 = x1 + b1 - int(x1 + b1)
        x2 = x2 + b2 - int(x2 + b2)
    else:
        x1 = np.random.uniform(0, 1) 
        x2 = np.random.uniform(0, 1) 
    data.append([x1, x2])
data_raw = np.array(data)
data = np.copy(data_raw)
np.random.shuffle(data)
print(data)
print()

#--- compute correl CI on M subsamples of size n, for various n

mu_x = np.zeros(M)
mu_y = np.zeros(M)
mu_xx = np.zeros(M)
mu_yy = np.zeros(M)
mu_xy = np.zeros(M)
correl = np.zeros(M)

for n in range(1, N): 

    for sample in range(M):

        idx = int(n + sample*N/M) % N
        obs = data[idx]
        mu_x[sample] += obs[0]
        mu_y[sample] += obs[1]
        mu_xx[sample] += obs[0]*obs[0]  
        mu_yy[sample] += obs[1]*obs[1]  
        mu_xy[sample] += obs[0]*obs[1]  

        if n > 1:  # otherwise variance is zero
            s_mu_x = mu_x[sample]/n  
            s_mu_y = mu_y[sample]/n  
            s_mu_xy = mu_xy[sample]/n 
            var_x = (mu_xx[sample]/n) - (s_mu_x*s_mu_x)
            var_y = (mu_yy[sample]/n) - (s_mu_y*s_mu_y)
            correl[sample] = (s_mu_xy - s_mu_x*s_mu_y) / np.sqrt(var_x * var_y)
  
    if n % 100 == 0:  
        print("Building CI for sample size", n)
        lower_bound = np.quantile(correl, p)
        upper_bound = np.quantile(correl, 1-p)
        CI_nobs.append(n)
        CI_mean.append(np.mean(correl)) 
        CI_lower.append(lower_bound)
        CI_upper.append(upper_bound)
        CI_width.append(upper_bound - lower_bound)

#--- Plot confidence intervals as a function of sample size

mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams['legend.fontsize'] = 0.1
plt.rc('xtick',labelsize=7)
plt.rc('ytick',labelsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.subplot(1, 2, 1)
plt.plot(CI_nobs, CI_mean, linewidth = 0.3)
plt.plot(CI_nobs, CI_lower, linewidth = 0.3)
plt.plot(CI_nobs, CI_upper, linewidth = 0.3)
plt.legend(['CI Center','CI Lower Bound','CI Upper Bound'],fontsize = 7)
plt.xlabel('Sample size for confidence interval (CI)', fontsize=7)
plt.ylabel('Estimated value for ρ',fontsize=7)

#--- Width of confidence interval based on sample size, and fitted curve

plt.subplot(1, 2, 2)
plt.plot(CI_nobs, CI_width, linewidth = 0.3)
plt.xlabel('Sample size for confidence interval (CI)', fontsize=7)
plt.ylabel('Width of confidence interval',fontsize=7)

CI_nobs_log = np.log(CI_nobs)
CI_width_log = np.log(CI_width)
table = np.stack((CI_nobs_log, CI_width_log), axis = 0)
cov = np.cov(table)
beta = cov[0, 1] / cov[0, 0]
log_alpha = np.mean(CI_width_log) - beta * np.mean(CI_nobs_log)
alpha = np.exp(log_alpha)
Fitted_width = alpha * (CI_nobs**beta)
print("alpha = ", alpha)
print("beta = ", beta)

plt.plot(CI_nobs, Fitted_width, linewidth = 0.3)
plt.ylim([0.00, 0.35])
plt.legend(['CI Width','Fitted curve'],fontsize = 7)
plt.show()

#--- Plot autocorrelations

nlags = 100
lags = np.arange(nlags)
acf_x_raw = sm.tsa.acf(data_raw[:,0], nlags = nlags)
acf_y_raw = sm.tsa.acf(data_raw[:,1], nlags = nlags)
acf_x = sm.tsa.acf(data[:,0], nlags = nlags)
acf_y = sm.tsa.acf(data[:,1], nlags = nlags)

plt.subplot(2,1,1)
plt.plot(lags[1:nlags], acf_x_raw[1:nlags], linewidth = 0.3)
plt.plot(lags[1:nlags], acf_y_raw[1:nlags], linewidth = 0.3)
plt.legend(['1st feature','2nd feature'],fontsize = 7, loc='upper right')
plt.xlabel('Autocorrelation function before reshuffling', fontsize=7)
plt.subplot(2,1,2)
plt.plot(lags[1:nlags], acf_x[1:nlags], linewidth = 0.3)
plt.plot(lags[1:nlags], acf_y[1:nlags], linewidth = 0.3)
plt.legend(['1st feature','2nd feature'],fontsize = 7, loc='upper right')
plt.xlabel('Autocorrelation function after reshuffling', fontsize=7)
plt.show()

plt.scatter(data[:,0], data[:,1], s = 0.1)
plt.show()
