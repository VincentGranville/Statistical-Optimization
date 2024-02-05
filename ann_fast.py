# Probabilistic ANN, can be used for clustering / classification

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from PIL import Image
import moviepy.video.io.ImageSequenceClip


#--- [1] Parameters and functions for visualizations

def save_image(fname,frame):

    # back-up function in case of problems with plt.savefig
    global fixedSize

    plt.savefig(fname, bbox_inches='tight')    
    # make sure each image has same size and size is multiple of 2
    # required to produce a viewable video   
    im = Image.open(fname)
    if frame == 0:  
        # fixedSize determined once for all in the first frame
        width, height = im.size
        width=2*int(width/2)
        height=2*int(height/2)
        fixedSize=(width,height)
    im = im.resize(fixedSize) 
    im.save(fname,"PNG")
    return()

def plot_frame():

    plt.scatter(x[:,0], x[:,1], color='red', s = 2.5) 
    z = []

    for k in range(N):

        neighbor = arr_NN[k]
        x_values = (x[k,0], y[neighbor,0]) 
        y_values = (x[k,1], y[neighbor,1]) 
        plt.plot(x_values,y_values,color='red',linewidth=0.1,marker=".",markersize=0.1) 
        z_obs = (y[neighbor,0], y[neighbor,1])
        z.append(z_obs)

    z = np.array(z)
    plt.scatter(y[:,0], y[:,1], s=10,  marker = '+', linewidths=0.5, color='green') 
    plt.scatter(z[:,0], z[:,1], s=10,  marker = '+', linewidths=0.5, color='blue')  
    return()

mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7


#--- [2] Create data, initial list of NN, and hash

def sort_x_by_NNdist_to_y(x, y, arr_NN):

    NNdist = {}
    x_tmp = np.copy(x)
    arr_NN_tmp = np.copy(arr_NN)
    for k in range(N):
        neighbor = arr_NN_tmp[k]
        NNdist[k] = np.sum(abs(x_tmp[k] - y[neighbor]))
    NNdist = dict(sorted(NNdist.items(), key=lambda item: item[1],reverse=True ))

    k = 0
    for key in NNdist:
        arr_NN[k] = arr_NN_tmp[key]
        x[k] = x_tmp[key]
        k += 1
    return(x, arr_NN)

seed = 57
np.random.seed(seed)
eps = 0.00000000001

N = 200            # number of points in x[]
K = int(0.5 * N)   # sort x[] by NN distance every K iterations
M = 200            # number of points in y[]

niter =  10000 
mean = [0, 0]
cov = [(0.1, 0),(0, 0.1)]
x = np.random.multivariate_normal(mean, cov, size=N)
y = np.random.multivariate_normal(mean, cov, size=M)
# y = np.copy(x) 
np.random.shuffle(x)
np.random.shuffle(y)

arr_NN = np.zeros(N)
arr_NN = arr_NN.astype(int)
hash = {}
sum_dist = 0

for k in range(N):

    # nearest neighbor to x[k] can't be identical to x[k]
    dist = 0

    while dist < eps:
       neighbor = int(np.random.randint(0, M))
       dist = np.sum(abs(x[k] - y[neighbor]))

    arr_NN[k] = neighbor
    sum_dist += np.sum(abs(x[k] - y[neighbor]))
    hash[k] = (-1,)

x, arr_NN = sort_x_by_NNdist_to_y(x, y, arr_NN)
low = sum_dist


#--- [3] Main part

mode     = 'minDist'  # options: 'minDist'  or 'maxDist'
optimize = 'speed'    # options: 'speed' or 'memory'
video    = False      # True if you want to produce a video
decay    = 0.0

history_val = []
history_arg = []
flist = []
swaps = 0
steps = 0
frame = 0

for iter in range(niter):

    k = iter % K 
    j = -1
    while j in hash[k] and len(hash[k]) <= N: 
        # if optimized for memory, there is always only iter in this loop
        steps += 1
        j = np.random.randint(0, M) # potential new neighbor y[j], to x[k]

    if optimize == 'speed':
        hash[k] = (*hash[k], j) 

    if len(hash[k]) <= N:

        # if optimized for memory, then len(hash[k]) <= N, always
        old_neighbor = arr_NN[k]
        new_neighbor = j
        old_dist = np.sum(abs(x[k] - y[old_neighbor]))
        new_dist = np.sum(abs(x[k] - y[new_neighbor]))
        if mode == 'minDist':
            ratio = new_dist/(old_dist + eps)
        else:
            ratio = old_dist/(new_dist + eps)
        if ratio < 1-decay/np.log(2+iter) and new_dist > eps: 
            swaps += 1
            arr_NN[k] = new_neighbor
            sum_dist += new_dist - old_dist
            if sum_dist < low:
                low = sum_dist

            if video and swaps % 4 == 0:

                fname='ann_frame'+str(frame)+'.png'
                flist.append(fname)
                plot_frame() 

                # save image: width must be a multiple of 2 pixels, all with same size
                # use save_image(fname,frame) in case of problems with plt.savefig
                plt.savefig(fname, dpi = 200)
                plt.close() 
                frame += 1

    if iter % K == K-1:
        x, arr_NN = sort_x_by_NNdist_to_y(x, y, arr_NN)

    if iter % 100 == 0:
        print("%6d %6d %6d %8.4f %8.4f" 
                 % (iter, swaps, steps, low/N, sum_dist/N))
        history_val.append(sum_dist/N)
        history_arg.append(steps) # try replacing steps by iter 


history_val = np.array(history_val)
history_arg = np.array(history_arg)

if video:
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(flist, fps=6)
    clip.write_videofile('ann.mp4')


#--- [4] Visualizations (other than the video)

plot_frame()
plt.show()

#- curve fitting for average NN distance (Y-axis) over time (X-axis)

# works only with mode == 'minDist'

def objective(x, a, b, c):
    return(a + b*(x**c)) 

# ignore first offset iterations, where fitting is poor
offset = 5

x = history_arg[offset:]
y = history_val[offset:]

# param_bounds to set bounts on curve fitting parameters
if mode == 'minDist':
    param_bounds=([0,0,-1],[np.inf,np.infty,0])  
else: 
    param_bounds=([0,0,0],[np.inf,np.infty,1])  

param, cov = curve_fit(objective, x, y, bounds = param_bounds)   
a, b, c = param
# is c = -1/2 the theoretical value, assuming a = 0?
print("\n",a, b, c)  

y_fit = objective(x, a, b, c)
## plt.plot(x, y, linewidth=0.4)
plt.plot(history_arg, history_val, linewidth=0.4)
plt.plot(x, y_fit, linewidth=0.4)
plt.legend(['Avg NN distance','Curve fitting'],fontsize = 7)
plt.show()
