# equantile.py: extrapolated quantiles, with video production

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from PIL import Image
import moviepy.video.io.ImageSequenceClip

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

data = get_test_data(100)
# data = get_real_data()
N = 1000000
truncate = False

# minz > maxz is the same as (minz = -infinity, maxz = +infinity)
if truncate == True:
    minz = 0.50 * np.min(data)  # use 0.95 for 'charges', 0.50 for 'bmi'
    maxz = 1.50 * np.max(data)  # use 1.50 for 'charges', 1.50 for 'bmi'
else:
    minz = 1.00
    maxz = 0.00

#--- Making video

mpl.rcParams['axes.linewidth'] = 0.3
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
bins=np.linspace(-7.0, 10.0, num=100)

pbins = 1000
step = N / pbins      # N must be a multiple of pbins
my_dpi = 300          # dots per each for images and videos 
width  = 2400         # image width
height = 1800         # image height
flist = []            # list image filenames for video
nframes = 500
velocity = 3.00

def save_image(fname,frame):
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

for frame in range(nframes):

    print("Processing frame", frame)
    v = 0.4*(frame/nframes)**velocity  ### np.log(1 + frame)/100
    sigma = v * np.std(data) 
    sample = mixture_deviate(N, data, truncated_norm, sigma, minz, maxz)
    equant = []

    for k in range(pbins):
        p = (k + 0.5) / pbins
        eq_index = int(step * (k + 0.5))
        equant.append(sample[eq_index])

    plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
    plt.hist(equant,color='orange',edgecolor='red',bins=bins,linewidth=0.3,label='v=%6.4f' %v)
    plt.legend(loc='upper right', prop={'size': 6}, )
    plt.ylim(0,60)
    fname='equant_frame'+str(frame)+'.png'
    flist.append(fname)
    save_image(fname,frame)
    plt.close()

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(flist, fps=10)
clip.write_videofile('equant.mp4')





