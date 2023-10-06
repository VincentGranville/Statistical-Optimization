import numpy as np

n = 5000
m = 10000
nu = 0.5
beta = 0.5

prange = np.zeros(m)
pmin = np.zeros(m)

for k in range(m):
    if k % 1000 == 0:
        print(k)
    x = np.random.uniform(0, 1, n)
    x = x**nu
    # x = np.random.triangular(0, 1, 2, n)
    # x = np.random.poisson(1, n)
    min = np.min(x)
    max = np.max(x)
    prange[k] = max - min
    pmin[k] = min

mean = np.mean(prange)
stdev = np.std(prange)   
mean2 = np.mean(pmin)
stdev2 = np.std(pmin)

print("\nRange expectation:    %6.5f\nMinimum expectation: %8.5f" %(mean, mean2))
print("Range stdev:          %6.5f\nMinimum stdev:       %8.5f" %
          ((n**beta)*stdev, (n**beta)*stdev2))  
