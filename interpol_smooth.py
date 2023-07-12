import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors  
from matplotlib import cm # color maps

n = 60         # number of observations in training set
ngroups = 3     # number of clusters in Gaussian mixture
seed = 13       # to initiate random number generator

#--- create locations for training set

width = 2 
height = 1
np.random.seed(seed)
x = np.random.uniform(0, width, n)
y = np.random.uniform(0, height, n)


#--- create values z attached to these locations

weights = np.random.uniform(0.5, 0.9, ngroups)
sum = np.sum(weights)
weights = weights/sum

cx = np.random.uniform(0, width, ngroups)
cy = np.random.uniform(0, height, ngroups)
sx = np.random.uniform(0.3, 0.4, ngroups)
sy = np.random.uniform(0.3, 0.4, ngroups)
rho = np.random.uniform(-0.3, 0.5, ngroups)

def f(x, y, cx, cy, sx, sy, rho):

    # bivariate bell curve

    tx = ( (x - cx) / sx)**2
    ty = ( (y - cy) / sy)**2
    txy = rho * (x - cx) * (y - cy) / (sx * sy)
    z = np.exp(-(tx - 2*txy + ty) / (2*(1 - rho**2)) )
    z = z / (sx * sy * np.sqrt(1 - rho**2))
    return(z)

def gm(x, y, weights, cx, cy, sx, sy, rho):

    # mixture of gaussians

    n = len(x)
    ngroups = len(cx)
    z = np.zeros(n)   
    for k in range(ngroups):
        z += weights[k] * f(x, y, cx[k], cy[k], sx[k], sy[k], rho[k])
    return(z)

z = gm(x, y, weights, cx, cy, sx, sy, rho)
npdata = np.column_stack((x, y, z))

print(npdata)

#--- model parameters

alpha = 1.0       # small alpha increases smoothing
beta  = 2.0       # small beta increases smoothing
kappa = 2.0       # high kappa makes method close to kriging 
eps   = 1.0e-8    # make it work if sample locations same as observed ones
delta = eps + 1.2 * max(width, height)   # don't use faraway points for interpolation

#--- interpolation for validation set: create locations

n_valid = 200   # number of locations to be interpolated, in validation set
xa = np.random.uniform(0, width, n_valid)
ya = np.random.uniform(0, height, n_valid)

#--- interpolation for validation set 

def w(x, y, x_k, y_k, alpha, beta):
    # distance function
    z = (abs(x - x_k)**beta + abs(y - y_k)**beta)**alpha
    return(z)

def interpolate(x, y, npdata, delta):

    # compute interpolated z at location (x, y) based on npdata (observations)
    # also returns npt, the number of data points used for each interpolated value
    # data points (x_k, y_k) with w[(x,y), (x_k,y_k)] >= delta are ignored
    # note: (x, y) can be a location or an array of locations

    if np.isscalar(x):  # transform scalar to 1-cell array
        x = [x]
        y = [y]
    sum  = np.zeros(len(x))
    sum_coeff = np.zeros(len(x)) 
    npt = np.zeros(len(x)) 
    
    for k in range(n):
        x_k = npdata[k, 0]
        y_k = npdata[k, 1]
        z_k = npdata[k, 2]
        coeff = 1
        for i in range(n):
            x_i = npdata[i, 0]
            y_i = npdata[i, 1]
            if i != k:
                numerator = w(x, y, x_i, y_i, alpha, beta)
                denominator = w(x_k, y_k, x_i, y_i, alpha, beta) 
                coeff *= numerator / (eps + denominator) 
        dist = w(x, y, x_k, y_k, alpha, beta)   
        coeff = (eps + dist)**(-kappa) * coeff / (1 + coeff) 
        coeff[dist > delta] = 0.0  
        sum_coeff += coeff
        npt[dist < delta] += 1   
        sum += z_k * coeff  
 
    z = sum / sum_coeff 
    return(z, npt)

(za, npt) = interpolate(xa, ya, npdata, 0.5*delta)

#--- inverse transform parameters (un-normalize) 

x_steps = 160  # to create grid with steps x steps points, to generate contours 
y_steps = 80 
xb = np.linspace(min(npdata[:,0])-0.50, max(npdata[:,0])+0.50, x_steps)
yb = np.linspace(min(npdata[:,1])-0.50, max(npdata[:,1])+0.50, y_steps)
xc, yc = np.meshgrid(xb, yb) 

#--- create grid and get interpolated values at grid locations
#    zgrid for interpolated values, zgrid_true for reak ones

zgrid = np.empty(shape=(x_steps,y_steps)) 
xg = []
yg = []
gmap = {}
idx = 0
for h in range(len(xb)):
    for k in range(len(yb)):
        xg.append(xb[h])
        yg.append(yb[k])
        gmap[h, k] = idx
        idx += 1 
z, npt = interpolate(xg, yg, npdata, 2.2*delta)

zgrid_true = np.empty(shape=(x_steps,y_steps))
xg = np.array(xg)
yg = np.array(yg)
z_true = gm(xg, yg, weights, cx, cy, sx, sy, rho) 

for h in range(len(xb)):
    for k in range(len(yb)):
        idx = gmap[h, k]
        zgrid[h, k] = z[idx]  
        zgrid_true[h, k] = z_true[idx]
zgridt = zgrid.transpose() 
zgridt_true = zgrid_true.transpose() 

#--- visualizations

nlevels = 20  # number of levels on contour plots

def set_plt_params():
    # initialize visualizations
    fig = plt.figure(figsize =(4, 3), dpi=200) 
    ax = fig.gca()
    plt.setp(ax.spines.values(), linewidth=0.1)
    ax.xaxis.set_tick_params(width=0.1)
    ax.yaxis.set_tick_params(width=0.1)
    ax.xaxis.set_tick_params(length=2)
    ax.yaxis.set_tick_params(length=2)
    ax.tick_params(axis='x', labelsize=4)
    ax.tick_params(axis='y', labelsize=4)
    plt.rc('xtick', labelsize=4) 
    plt.rc('ytick', labelsize=4) 
    plt.rcParams['axes.linewidth'] = 0.1
    return(fig,ax)

# contour plot, interpolated values, full grid

(fig1, ax1) = set_plt_params() 
cs1 = plt.contourf(xc, yc, zgridt, cmap='coolwarm',levels=nlevels,linewidths=0.1)  
sc1 = plt.scatter(npdata[:,0], npdata[:,1], c=npdata[:,2], s=8, cmap=cm.coolwarm,
      edgecolors='black',linewidth=0.3,alpha=0.8)
cbar1 = plt.colorbar(sc1)
cbar1.ax.tick_params(width=0.1) 
cbar1.ax.tick_params(length=2) 
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()
           
# scatter plot: validation set (+) and training data (o)         

(fig2, ax2) = set_plt_params()
my_cmap = mpl.colormaps['coolwarm']  # old version: cm.get_cmap('coolwarm')
my_norm = colors.Normalize()
ec_colors = my_cmap(my_norm(npdata[:,2]))
sc2a = plt.scatter(npdata[:,0], npdata[:,1], c='white', s=5, cmap=my_cmap, 
    edgecolors=ec_colors,linewidth=0.4)
sc2b = plt.scatter(xa, ya, c=za, cmap=my_cmap, marker='+',s=5,linewidth=0.4)
plt.show()
plt.close()

#--- measuring quality of the fit on validation set
#    zd is true value, za is interpolated value

zd = gm(xa, ya, weights, cx, cy, sx, sy, rho) 
error = np.average(abs(zd - za)) 
print("\nMean absolute error on validation set: %6.2f" %(error))
print("Mean value on validation set: %6.2f" %(np.average(zd)))

#--- plot of original function (true values)

(fig3, ax3) = set_plt_params() 
cs3 = plt.contourf(xc, yc, zgridt_true, cmap='coolwarm',levels=nlevels,linewidths=0.1)
cbar1 = plt.colorbar(cs3)
cbar1.ax.tick_params(width=0.1) 
cbar1.ax.tick_params(length=2) 
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()

#--- compute smoothness of interpolated grid via double gradient 
#    1/x_steps and 1/y_steps are x, y increments between 2 adjacent grid locations

h2 = x_steps**2 
k2 = y_steps**2
dx, dy = np.gradient(zgrid)    # zgrid_true for original function
zgrid_norm1 = np.sqrt(h2*dx*dx + k2*dy*dy)
dx, dy = np.gradient(zgrid_norm1)    
zgrid_norm2 = np.sqrt(h2*dx*dx + k2*dy*dy)  
zgridt_norm2 = zgrid_norm2.transpose()
average_smoothness = np.average(zgrid_norm2) 
print("Average smoothness of interpolated grid: %6.3f" %(average_smoothness)) 

(fig4, ax4) = set_plt_params() 
cs4 = plt.contourf(xc, yc, zgridt_norm2, cmap=my_cmap,levels=nlevels,linewidths=0.1)
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()
