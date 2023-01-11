import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# https://www.digitalocean.ie/Data/DownloadTideData

mode = 'Data' # options: 'Data', 'Math'
 
IN = open("tides_Dublin.txt","r")
table = IN.readlines()
IN.close()

temp={}
t = 0     
for string in table: 
    string = string.replace('\n', '')
    fields = string.split('\t')
    temp[t/16]=float(fields[0])
    t = t + 1
nobs = len(temp)

def g(t):
    if mode == 'Data':
        z = temp[t]
    else:
        z = 1.5 + np.cos(0.319*t-0.871)-0.7*np.sin(0.654*t+2.343) 
    return(z)

def interpolate(t, eps): 
    sum = 0
    t_0 = int(t + 0.5) # closest interpolation node to t
    pi2 = 2/np.pi  
    flag1 = -1  
    flag2 = -1  
    for k in range(0, n):
        # use nodes k1, k2 in interpolation formula
        k1 = t_0 + k
        k2 = t_0 - k
        tt = t - t_0
        if k != 0: 
            if k %2 == 0:
                z = g(k1) + g(k2) 
                if abs(tt**2 - k**2) > eps:
                    term = flag1 * tt*z*pi2 * np.sin(tt/pi2) / (tt**2 - k**2)
                else:    
                    # use limit as tt --> k
                    term = z/2
                flag1 = -flag1
            else: 
                z = g(k1) - g(k2) 
                if abs(tt**2 - k**2) > eps:
                    term = flag2 * tt*z*pi2 * np.cos(tt/pi2) / (tt**2 - k**2)
                else: 
                    # use limit as tt --> k
                    term = z/2
                flag2 = -flag2
        else: 
            z = g(k1)
            if abs(tt) > eps:
                term = z*pi2*np.sin(tt/pi2) / tt
            else:
                # use limit as tt --> k (here k = 0)
                term = z
        sum += term
    return(sum)

#--- main loop

t_min  = 120    # interpolate between t_min and t_max
t_max  = 180    # interpolate between t_min and t_max
incr   = 1/16    # time increment between nodes
n      = 8      # 2n+1 is number of nodes used in interpolation 
eps    = 1.0e-12 

OUT = open("interpol_tides_Dublin.txt","w")

time = []
ze = []
zi = []

fig = plt.figure(figsize=(6,3))
mpl.rcParams['axes.linewidth'] = 0.1
mpl.rc('xtick', labelsize=6) 
mpl.rc('ytick', labelsize=6) 

for t in np.arange(t_min, t_max, incr):  
    time.append(t)
    z_interpol = interpolate(t, eps) 
    z_exact = g(t)
    zi.append(z_interpol)
    ze.append(z_exact)
    error = abs(z_exact - z_interpol)
    if error > 0:
        plt.scatter(t,1+error,color='black', marker='.',s=0.05)
    if t == int(t):
        plt.scatter(t,z_exact,color='red', s=6)
    OUT.write("%10.6f\t%10.6f\t%10.6f\t%10.6f\n" % (t,z_exact,z_interpol,error))
OUT.close()

plt.plot(time,ze,color='red',linewidth = 1.0, alpha=0.5) ### , s=0.1)
plt.plot(time,zi,color='blue', linewidth = 1.0,alpha=0.5) ##, s=0.1)
plt.savefig('tides2.png', dpi=300)
plt.show()
