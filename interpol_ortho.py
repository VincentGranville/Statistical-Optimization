import numpy as np
import random

#---- make up data

data = []
nobs = 100 
random.seed(69)
for i in range(nobs):
    x1 = -1 + 2*random.random()           # feature 1
    x2 = -1 + 2*random.random()           # feature 2
    z  = np.sin(0.56*x1) - 0.5*np.cos(1.53*x2)  # response
    obs = [x1, x2, z]
    data.append(obs)

npdata = np.array(data)
transf_npdata = npdata  # no data transformer needed here

#--- the p_k functions 

def p_k(x, k):

    # if input x is an array, output z is also an array

    if k % 2 == 0:
        z = np.cos(k*x*np.pi)
    else: 
        z = np.sin(k*x*np.pi)
    return(z)

#--- beta_k, alpha_k, gamma_k coefficients

intercept = np.ones(nobs)   
p_0 = p_k(intercept, k = 0)
p_1 = p_k(transf_npdata[:,0], k = 1)  # feature 1
p_2 = p_k(transf_npdata[:,1], k = 2)  # feature 2 

gamma_0 = np.dot(p_0, p_0) # dot product
gamma_1 = np.dot(p_1, p_1)
gamma_2 = np.dot(p_2, p_2)

observed_temp =  npdata[:,2] 
beta_0 = np.dot(p_0, observed_temp)
beta_1 = np.dot(p_1, observed_temp)
beta_2 = np.dot(p_2, observed_temp)

alpha_0 = beta_0 / gamma_0
alpha_1 = beta_1 / gamma_1
alpha_2 = beta_2 / gamma_2

#--- interpolation 

predicted_temp = alpha_0 * p_0 + alpha_1 * p_1 + alpha_2 * p_2 

#--- print results: predicted vs observed

for i in range(nobs):
    print("%8.5f %8.5f" %(predicted_temp[i],observed_temp[i]))

correlmatrix = np.corrcoef(predicted_temp,observed_temp)
correlation = correlmatrix[0, 1]
print("corr between predicted/observed: %8.5f" % (correlation))

#--- interpolate for new observation (with intercept = 1)

x1 =  0.234
x2 = -0.541

z_predicted = alpha_0 * p_k(1,k=0) + alpha_1 * p_k(x1,k=1) + alpha_2 * p_k(x2,k=2)
print("test interpolation: z_predict = %8.5f" %(z_predicted))
 
