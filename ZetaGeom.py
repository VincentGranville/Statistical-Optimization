import numpy as np

#--- compute mean and stdev of ZetaGeom[p, a]

def ZetaGeom(p, a):
    C = 0
    for k in range(200):
        C += p**k/(k+1)**a
    mu = 0
    m2 = 0
    for k in range(200):
        mu += k*p**k/(k+1)**a
        m2 += k*k*p**k/(k+1)**a
    mu /= C
    m2 /= C
    var = m2 - mu*mu
    stdev = var**(1/2)
    return(mu, stdev)

#--- optimized grid search to find optimal p and a

def grid_search(grid_range):
    p_min = grid_range[0][0]
    p_max = grid_range[0][1]
    a_min = grid_range[1][0]
    a_max = grid_range[1][1]
    p_step = (p_max - p_min)/10
    a_step = (a_max - a_min)/10
    min_delta = 999999999.9
    for p in np.arange(p_min, p_max, p_step):
        for a in np.arange(a_min, a_max, a_step):
            (mu, std) = ZetaGeom(p, a)
            delta = np.sqrt((mu - target_mu)**2 + (std - target_std)**2)
            if delta < min_delta:
                p_best = p
                a_best = a
                mu_best = mu
                std_best = std
                min_delta = delta
    return(p_best, a_best, mu_best, std_best, min_delta)

#--- estimating p and a based on observed mean and standard deviation

target_mu    = 1.095  # mean
target_std   = 1.205  # standard deviation

p = 0.5
a = 0.0
step_p = 0.4
step_a = 3.0

for level in range(3):
    step_p /= 2
    step_a /= 2
    p_min = max(0, p - step_p)
    p_max = p + step_p
    a_min = a - step_a
    a_max = a + step_a
    grid_range = [(p_min, p_max),(a_min, a_max)]
    (p, a, mu, std, min_delta) = grid_search(grid_range)
    print("delta: %6.4f mu: %6.4f std: %6.4f p: %6.4f a: %6.4f" 
         % (min_delta, mu, std, p, a))

# now (p_fit, a_fit) is such that (mean, std) = (target_mu, target_std)
p_fit = p  
a_fit = a

# now we found the correct p, a to fit to target_mu, target stdev

#--- sampling from ZetaGeom[p, a]

def CDF(p, a):
    C = 0
    for k in range(100):
        C += p**k/(k+1)**a
    arr_CDF = []
    CDF = 0
    for k in range(100):
        CDF += (p**k/(k+1)**a)/C
        arr_CDF.append(CDF)
    return(arr_CDF)

def sample_from_CDF(p, a):
    u = np.random.uniform(0,1)
    k = 0
    arr_CDF = CDF(p, a)
    while u > arr_CDF[k]:
        k = k+1
    return(k)

#--- sample using estimated p, a to match target mean and stdev

nobs = 50000  # number of deviates to produce
seed = 500
np.random.seed(seed)
sample1 = np.empty(nobs)
for n in range(nobs):
    k = sample_from_CDF(p_fit, a_fit)
    sample1[n] = k

mean = np.mean(sample1)
std  = np.std(sample1)
maxx = max(sample1)
print("\nSample stats: mean: %5.3f std: %5.3f max: %5.3f" 
   % (mean, std, maxx))

#--- optional: plotting approximation error for p, a 

from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm # color maps

xa = np.arange(0.0, 0.6, 0.005)
ya = np.arange(-3.0, 0.0, 0.025)
xa, ya = np.meshgrid(xa, ya)
za = np.empty(shape=(len(xa),len(ya)))

kk = 0
for p in np.arange(0.0, 0.6, 0.005):
    hh = 0
    for a in np.arange(-3.0, 0.0, 0.025):
        (mu, std) = ZetaGeom(p, a)
        delta = np.sqrt((mu - target_mu)**2 + (std - target_std)**2)
        za[hh, kk] = delta
        hh += 1
    kk += 1

mpl.rcParams['axes.linewidth'] = 0.5
fig = plt.figure() 
axes = plt.axes()
axes.tick_params(axis='both', which='major', labelsize=8)
axes.tick_params(axis='both', which='minor', labelsize=8)
CS = axes.contour(xa, ya, za, levels=150, cmap=cm.coolwarm, linewidths=0.35)
cbar = fig.colorbar(CS, ax = axes, shrink = 0.8, aspect = 5)
cbar.ax.tick_params(labelsize=8)
plt.show()

#--- compare with zeta with same mean mu = 1.095

p = 1.0
a = 2.33  # a < 3 thus var is infinite
sample2 = np.empty(nobs)
for n in range(nobs):
    k = sample_from_CDF(p, a)
    sample2[n] = k

mean = np.mean(sample2)
std  = np.std(sample2)
maxx = max(sample2)
print("Sample stats Zeta: mean: %5.3f std: %5.3f max: %5.3f" 
   % (mean, std, maxx))

#--- compare with geom with same mean mu = 1.095

p = target_mu/(1 + target_mu)
a = 0.0
sample3 = np.empty(nobs)
for n in range(nobs):
    k = sample_from_CDF(p, a)
    sample3[n] = k

mean = np.mean(sample3)
std  = np.std(sample3)
maxx = max(sample3)
print("Sample stats Geom: mean: %5.3f std: %5.3f max: %5.3f" 
   % (mean, std, maxx))

#--- plot probability density functions

axes.tick_params(axis='both', which='major', labelsize=4)
axes.tick_params(axis='both', which='minor', labelsize=4)
mpl.rc('xtick', labelsize=8) 
mpl.rc('ytick', labelsize=8) 
plt.xlim(-0.5,9.5)
plt.ylim(0,0.8)

cdf1 = CDF(p_fit, a_fit) 
cdf2 = CDF(1.0, 2.33) 
cdf3 = CDF(target_mu/(1+target_mu), 0.0) 

for k in range(10):
    if k == 0:
        pdf1 = cdf1[0]
        pdf2 = cdf2[0]
        pdf3 = cdf3[0]
    else:
        pdf1 = cdf1[k] - cdf1[k-1]
        pdf2 = cdf2[k] - cdf2[k-1]
        pdf3 = cdf3[k] - cdf3[k-1]
    plt.xticks(np.linspace(0,9,num=10))
    plt.plot([k+0.2,k+0.2],[0,pdf1],linewidth=5, c='tab:green', label='Zeta-geom')
    plt.plot([k-0.2,k-0.2],[0,pdf2],linewidth=5, c='tab:orange',label='Zeta')
    plt.plot([k,k],[0,pdf3],linewidth=5, c='tab:gray',label='Geom')

plt.legend(['Zeta-geom','Zeta','Geom'],fontsize = 7)
plt.show()
