import numpy as np

n = 100
eps = 1.0e-8

def g(t): 
    alpha = 1.4    # |a| < pi/2 
    z = np.sin(alpha*t)/(t)
    return(z)

def interpolate(t): 
    sum = 0
    flag1 = -1
    flag2 = -1
    for k in range(1,n):
        if k%2 == 0:
            z = g(k) + g(-k) 
            term = flag1 * np.sin(t*np.pi/2) 
            flag1 = -flag1
        else: 
            z = g(k) - g(-k) 
            term = flag2 * np.cos(t*np.pi/2) 
            flag2 = -flag2
        term = term*t*z/(t**2 - k**2)
        sum += term
    sum = (2/np.pi)*(sum + np.sin(t*np.pi/2)*g(0 + eps)/t)
    return(sum)

for t in np.arange(-3, 3+eps, 0.1):  
    z_exact = g(t + eps)
    z_interpol = interpolate(t + eps) 
    error = abs(z_exact - z_interpol)
    print("%10.6f %10.6f %10.6f %10.6f" % (t,z_exact,z_interpol,error))
