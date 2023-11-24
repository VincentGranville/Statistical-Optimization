# equantile.py: extrapolated quantiles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

seed = 76
np.random.seed(seed)

def get_test_data(n=100):
    data = []
    for k in range(n):
        u = np.random.uniform(0, 1)
        if u < 0.2:
            x = np.random.normal(-1, 1)
        elif u < 0.7:
            x = np.random.normal(0, 2)
        else: 
            x = np.random.normal(5.5, 0.8)
        data.append(x)
    data = np.array(data)
    return(data)

def get_real_data():
    url = "https://raw.githubusercontent.com/VincentGranville/Main/main/insurance.csv"
    data = pd.read_csv(url)
    # features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'] 
    data = data['bmi']    # choose 'bmi' or 'charges'
    data = np.array(data)
    return(data)

#--

def truncated_norm(mu, sigma, minz, maxz):
    z = np.random.normal(mu, sigma)
    if minz < maxz:
        while z < minz or z > maxz:
            z = np.random.normal(mu, sigma)
    return(z)

#- sample from mixture

def mixture_deviate(N, data, f, sigma, minz, maxz, verbose=False):
    sample = []
    point_idx = np.random.randint(0, len(data), N) 
    mu = data[point_idx]
    for k in range(N):
        z = truncated_norm(mu[k], sigma, minz, maxz)
        sample.append(z)
        if verbose and k%10 == 0:
            print("sampling %6d / %6d" %(k, N))
    sample = np.array(sample)
    sample = np.sort(sample)
    return(sample)

#--- Main part

# data = get_test_data(100)
data = get_real_data()
N = 1000000
truncate = False

# minz > maxz is the same as (minz = -infinity, maxz = +infinity)
if truncate == True:
    minz = 0.50 * np.min(data)  # use 0.95 for 'charges', 0.50 for 'bmi'
    maxz = 1.50 * np.max(data)  # use 1.50 for 'charges', 1.50 for 'bmi'
else:
    minz = 1.00
    maxz = 0.00

sigma1 = 0.0 * np.std(data) 
sample1 = mixture_deviate(N, data, truncated_norm, sigma1, minz, maxz)

sigma2 = 0.1 * np.std(data) 
sample2 = mixture_deviate(N, data, truncated_norm, sigma2, minz, maxz)

sigma3 = 0.2 * np.std(data) 
sample3 = mixture_deviate(N, data, truncated_norm, sigma3, minz, maxz)

sigma4 = 0.4 * np.std(data) 
sample4 = mixture_deviate(N, data, truncated_norm, sigma4, minz, maxz)

arrq = []
equant1 = []
equant2 = []
equant3 = []
equant4 = []
pquant = []

pbins = 1000
step = N / pbins    # N must be a multiple of pbins
for k in range(pbins):
    p = (k + 0.5) / pbins
    arrq.append(p)
    eq_index = int(step * (k + 0.5))
    equant1.append(sample1[eq_index])
    equant2.append(sample2[eq_index])
    equant3.append(sample3[eq_index])
    equant4.append(sample4[eq_index])
    pquant.append(np.quantile(data, p))

mpl.rcParams['axes.linewidth'] = 0.3
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

#--- Plot results

bins=np.linspace(np.min(equant4), np.max(equant4), num=100)

plt.subplot(2,2,1)
plt.hist(equant1,color='orange',edgecolor='red',bins=bins,linewidth=0.3,label='v=0.0')
plt.legend(loc='upper right', prop={'size': 6}, )
plt.ylim(0,35)
plt.subplot(2,2,2)
plt.hist(equant2,color='orange',edgecolor='red',bins=bins,linewidth=0.3,label='v=0.1')
plt.legend(loc='upper right', prop={'size': 6}, )
plt.ylim(0,35)
plt.subplot(2,2,3)
plt.hist(equant3,color='orange',edgecolor='red',bins=bins,linewidth=0.3,label='v=0.2')
plt.legend(loc='upper right', prop={'size': 6}, )
plt.ylim(0,35)
plt.subplot(2,2,4)
plt.hist(equant4,color='orange',edgecolor='red',bins=bins,linewidth=0.3,label='v=0.4')
plt.legend(loc='upper right', prop={'size': 6}, )
plt.ylim(0,35)
plt.show()

#--- Output some summary stats

print()
print("Observation range, min: %8.2f" %(np.min(data)))
print("Observation range, max: %8.2f" %(np.max(data)))
pmin = np.quantile(data, 0.5/pbins)
pmax = np.quantile(data, 1 - 0.5/pbins)
print("Python quantile %6.4f: %8.2f" % (0.5/pbins, pmin))
print("Python quantile %6.4f: %8.2f" % (1-0.5/pbins, pmax))
print("Python quantile %6.4f: %8.2f" % (0.5, np.quantile(data,0.5)))
print("Dataset stdev         : %8.2f" %(np.std(data)))

print()
print("sigma1: %6.2f" %(sigma1))
print("Equant quantile %6.4f: %8.2f" %(0.5/pbins, equant1[0]))
print("Equant quantile %6.4f: %8.2f" %(1-0.5/pbins, equant1[999]))
print("Equant quantile %6.4f: %8.2f" %(0.5, np.median(equant1)))
print("Equant-based stdev    : %8.2f" %(np.std(equant1)))

print()
print("sigma2: %6.2f" %(sigma2))
print("Equant quantile %6.4f: %8.2f" %(0.5/pbins, equant2[0]))
print("Equant quantile %6.4f: %8.2f" %(1-0.5/pbins, equant2[999]))
print("Equant quantile %6.4f: %8.2f" %(0.5, np.median(equant2)))
print("Equant-based stdev    : %8.2f" %(np.std(equant2)))

print()
print("sigma3: %6.2f" %(sigma3))
print("Equant quantile %6.4f: %8.2f" %(0.5/pbins, equant3[0]))
print("Equant quantile %6.4f: %8.2f" %(1-0.5/pbins, equant3[999]))
print("Equant quantile %6.4f: %8.2f" %(0.5, np.median(equant3)))
print("Equant-based stdev    : %8.2f" %(np.std(equant3)))

print()
print("sigma4: %6.2f" %(sigma4))
print("Equant quantile %6.4f: %8.2f" %(0.5/pbins, equant4[0]))
print("Equant quantile %6.4f: %8.2f" %(1-0.5/pbins, equant4[999]))
print("Equant quantile %6.4f: %8.2f" %(0.5, np.median(equant4)))
print("Equant-based stdev    : %8.2f" %(np.std(equant4)))

