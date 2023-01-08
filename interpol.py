import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors  
from matplotlib import cm # color maps

data = [
# (latitute, longitude, temperature)
# source = https://cybergisxhub.cigi.illinois.edu/notebook/spatial-interpolation/
(41.878377,-87.627678,28.24),
(41.751238,-87.712990,19.83),
(41.736314,-87.624179,26.17),
(41.722457,-87.575350,45.70),
(41.736495,-87.614529,35.07),
(41.751295,-87.605288,36.47),
(41.923996,-87.761072,22.45),
(41.866786,-87.666306,45.01), # 125.01 outlier changed to 45.01
(41.808594,-87.665048,19.82),
(41.786756,-87.664343,26.21),
(41.791329,-87.598677,22.04),
(41.751142,-87.712990,20.20),
(41.831070,-87.617298,20.50),
(41.788979,-87.597995,42.15),
(41.914094,-87.683022,21.67),
(41.871480,-87.676440,25.14),
(41.736593,-87.604759,45.01), # 125.01 outlier changed to 45.01
(41.896157,-87.662391,21.16),
(41.788608,-87.598713,19.50),
(41.924903,-87.687703,21.61),
(41.895005,-87.745817,32.03),
(41.892003,-87.611643,28.30),
(41.839066,-87.665685,20.11),
(41.967590,-87.762570,40.60),
(41.885750,-87.629690,42.80),
(41.714021,-87.659612,31.46),
(41.721301,-87.662630,21.35),
(41.692703,-87.621020,21.99),
(41.691803,-87.663723,21.62),
(41.779744,-87.654487,20.88),
(41.820972,-87.802435,20.55),
(41.792543,-87.600008,20.41)
]
npdata = np.array(data)

#--- top parameters 

n = len(npdata)   # number of points in data set
ppo = 4           # create ppo new points around each observed point
new_obs = n * ppo  
alpha = 1.0       # small alpha increases smoothing
beta  = 2.0       # small beta increases smoothing
kappa = 2.0       # high kappa makes method close to kriging
eps   = 1.0e-8    # make it work if sample locations same as observed ones
np.random.seed(6)
radius = 1.2
audit  = True     # so log monitoring info about the interpolation        

xa = []                 # latitute
ya = []                 # longitude
da = []                 # dist between observed and interpolated value
zd = []                 # observed z
za = np.empty(new_obs)  # interpolated z

#--- transform data: normalization 

mu = npdata.mean(axis=0)
stdev = npdata.std(axis=0)
npdata = (npdata - mu)/stdev

#--- interpolation for sampled locations

def w(x, y, x_k, y_k, alpha, beta):
    # distance function
    z = (abs(x - x_k)**beta + abs(y - y_k)**beta)**alpha
    return(z)

# create random locations for interpolation purposes  
for h in range(ppo):
    # sample points in a circle of radius "radius" around each obs 
    xa = np.append(xa, npdata[:,0] + radius * np.random.uniform(-1, 1, n))
    ya = np.append(ya, npdata[:,1] + radius * np.random.uniform(-1, 1, n))
    da = np.append(da, w(xa[-n:],ya[-n:],npdata[:,0],npdata[:,1],alpha,beta))
    zd = np.append(zd, npdata[:,2])

delta = eps + max(da)   # to ignore obs too far away from sampled point
npt = np.empty(new_obs) # number of points used for interpolation at location j

def interpolate(x, y, npdata, delta, audit):
    # compute interpolated z at location (x, y) based on npdata (observations)
    # also returns npoints, the number of data points used in the interpolation
    # data points (x_k, y_k) with w[(x,y), (x_k,y_k)] >= delta are ignored
    # note: (x, y) can be a location or an array of locations

    sum  = 0.0
    sum_coeff = 0.0
    npoints = 0
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
        if dist < delta:
            coeff = (eps + dist)**(-kappa) * coeff / (1 + coeff) 
            sum_coeff += coeff
            npoints += 1
            if audit:
                OUT.write("%3d\t%3d\t%8.5f\t%8.5f\t%8.5f\n" % (j,k,z_k,coeff,dist))
        else:
            coeff = 0.0
        sum += z_k * coeff  
    if npoints > 0:
        z = sum / sum_coeff 
    else:
        z = 'NaN'  # undefined
    return(z, npoints)

OUT=open("audit.txt","w")   # output file for auditing / detecting issues
OUT.write("j\tk\tz_k\tcoeff\tdist\n")

for j in range(new_obs):
    (za[j], npt[j]) = interpolate(xa[j], ya[j], npdata, 0.5*delta, audit=True)

OUT.close()

#--- inverse transform (un-normalize) and visualizations

steps = 140  # to create grid with steps x steps points, to generate contours
xb = np.linspace(min(npdata[:,0])-0.50, max(npdata[:,0])+0.50, steps)
yb = np.linspace(min(npdata[:,1])-0.50, max(npdata[:,1])+0.50, steps)
xc = mu[0] + stdev[0] * xb
yc = mu[1] + stdev[1] * yb
xc, yc = np.meshgrid(xc, yc)
zgrid = np.empty(shape=(len(xb),len(yb)))   

# create grid and get interpolated values at grid locations
for h in range(len(xb)):
    for k in range(len(yb)):
        x = xb[h]
        y = yb[k]
        (z, points) = interpolate(x, y, npdata, 2.2*delta, audit=False)
        if z == 'NaN':
            zgrid[h,k] = 'NaN'
        else: 
            zgrid[h,k] = mu[2] + stdev[2] * z
zgridt = zgrid.transpose()

# inverse transform
xa = mu[0] + stdev[0] * xa
ya = mu[1] + stdev[1] * ya
za = mu[2] + stdev[2] * za
xb = mu[0] + stdev[0] * xb
yb = mu[1] + stdev[1] * yb
npdata = mu + stdev * npdata

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

# contour plot
(fig, ax) = set_plt_params() 
cs = plt.contourf(yc, xc, zgridt,cmap='coolwarm',levels=16) 
cbar = plt.colorbar(cs)
cbar.ax.tick_params(width=0.1) 
cbar.ax.tick_params(length=2) 
plt.scatter(npdata[:,1], npdata[:,0], c=npdata[:,2], s=8, cmap=cm.coolwarm,
      edgecolors='black',linewidth=0.3,alpha=0.8)
plt.show()
plt.close()
           
# scatter plot        
(fig, ax) = set_plt_params()
my_cmap = cm.get_cmap('coolwarm')
my_norm = colors.Normalize()
ec_colors = my_cmap(my_norm(npdata[:,2]))
plt.scatter(npdata[:,1], npdata[:,0], c='white', s=5, cmap=cm.coolwarm,
    edgecolors=ec_colors,linewidth=0.4)
sc=plt.scatter(ya[npt>0], xa[npt>0], c=za[npt>0], cmap=cm.coolwarm, 
    marker='+',s=5,linewidth=0.4) 

# show in green points not interpolated as they were too far away
plt.scatter(ya[npt==0], xa[npt==0], c='lightgreen', marker='+', s=5, 
    linewidth=0.4) 

cbar = plt.colorbar(sc)
cbar.ax.tick_params(width=0.1) 
cbar.ax.tick_params(length=2)
# plt.ylim(min(npdata[:,0]),max(npdata[:,0]))
# plt.xlim(min(npdata[:,1]),max(npdata[:,1]))
plt.show()

#--- measuring quality of the fit

error = np.mean(abs(za[npt>0] - zd[npt>0]))
print("Error=",delta)
