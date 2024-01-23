import numpy as np
from collections import OrderedDict

#--- [1] Functions to generate random digits

def get_next_digit(x, p, q, tau, step, base, option = "Home-Made"):

    if option == "Numpy":
        digit = np.random.randint(0, base)
    elif option == "Home-Made":
        x =  ((p * x) // q) % tau  # integer division for big int
        tau += step
        digit = x % base
    return(digit, x, tau)


def update_runs(digit, old_digit, run, max_run, hash_runs):

    if digit == old_digit:
        run += 1
    else:
        if (old_digit, run) in hash_runs:
            hash_runs[(old_digit, run)] += 1
        else:
            hash_runs[(old_digit, run)] = 1
        if run > max_run:
            max_run = run
        run = 1
    return(run, max_run, hash_runs)


def update_blocks(digit, m, block, base, block_size, hash_blocks):

    if m < block_size:
        block = base * block + digit
        m += 1
    else:
        if block in hash_blocks:
            hash_blocks[block] += 1
        else:
            hash_blocks[block] = 1
        block = 0
        m = 0
    return(m, block, hash_blocks)


def update_cp(digit, k, buffer, max_lag, N, cp_data):

    # processing k-th digit starting at k=2
    # buffer stores the last max_lag digits
    # cp stands for the cross-product part (in autocorrelation)

    mu = cp_data[0]
    cnt = cp_data[1]
    cp_vals = cp_data[2]
    cp_cnt = cp_data[3]

    buffer[(k-2) % max_lag] = digit
    mu += digit
    cnt += 1

    for lag in range(max_lag):
        if k-2 >= lag: 
            cp_vals[lag] += digit * buffer[(k-2-lag) % max_lag]
            cp_cnt[lag] += 1

    cp_data = (mu, cnt, cp_vals, cp_cnt)
    return(cp_data, buffer)


def generate_digits(N, x0, p, q, tau, step, base, block_size, max_lag, option = "Home-Made"):

    # Main function. Also produces output to test randomnes:
    #     - hash_runs and max_run: input for the run_test function
    #     - hash_blocks: input for the block_test function
    #     - cp_data input for the correl_test function

    # for run and block test
    hash_runs = {}
    hash_blocks = {}
    block = 0 
    m = 0
    run = 0
    max_run = 0
    digit = -1    
 
    # for correl_test
    mu = 0 
    cnt = 0 
    buffer = np.zeros(max_lag) 
    cp_vals = np.zeros(max_lag) # cross-products for autocorrel 
    cp_cnt = np.zeros(max_lag) 
    cp_data = (mu, cnt, cp_vals, cp_cnt)

    x = x0

    for k in range(2, N): 

        old_digit = digit
        (digit, x, tau) = get_next_digit(x, p, q, tau, step, base, option) 
        (run, max_run, hash_runs) = update_runs(digit, old_digit, run, max_run, hash_runs)
        (m, block, hash_blocks) = update_blocks(digit, m, block, base, block_size, hash_blocks)
        (cp_data, buffer) = update_cp(digit, k, buffer, max_lag, N, cp_data)

    print("----------------")        
    print("PRNG = ", option) 
    print("block_size (digits per block), digit base: %d, %d", block_size, base)
    if option == "Home-Made":
        print("p, q: %d, %d" %(p, q))
        print(len(str(x)), "decimal digits in last x")
    return(hash_runs, hash_blocks, max_run, cp_data)


#--- [2] Functions to perform tests of randomness

def run_test(base, max_run, hash_runs):

    # For each run, chi2 has approx. chi2 distrib. with base degrees of freedom
    # This is true assuming the digits are random

    print()
    print("Digit  ", end = " ")
    if base <= 8:
        for digit in range(base):
            print("%8d" %(digit), end =" ")
    print("     Exp", end = " ")
    print("     Avg", end = " ") # count average over all digits 
    print("    Chi2", end = " ") # degrees of freedom = base
    print("    norm", end = " ") 
    print("\n")

    for run in range(1, max_run+1):

        print("Run %3d" % (run), end = " ")
        prob = ((base-1)/base)**2  * (1/base)**run 
        exp  = N*prob 
        var  = N*prob*(1-prob)
        avg  = 0
        chi2 = 0  

        for digit in range(0, base):
            key = (digit, run)
            count = 0
            if key in hash_runs:
                count = hash_runs[key]
                avg += count
                chi2 += (count - exp)**2 / var 
            if base <= 8: 
                print("%8d" %(count), end =" ")

        avg /= base
        norm = (chi2 - base) / (2*base)
        print("%8d" %(int(0.5 + exp)), end =" ")
        print("%8d" %(int(0.5 + avg)), end =" ")
        print("%8.2f" %chi2, end =" ")
        print("%8.4f" %norm, end =" ")
        print()

    return()


def get_closest_blocks(block, list, trials):

    # Used in block_test
    # Return (left_block, right_block) with left_block <= block <= right_block, 
    #   - If block is in list, left_block = block = right_block
    #   - Otherwise, left_block = list(left), right_block = list(right); 
    #                they are the closest neighbors to block, found in list
    #   - left_block, right_block are found in list with weighted binary search
    #   - list must be ordered
    # trials: to compare spped of binary search with weighted binary search

    found = False
    left = 0
    right = len(list) - 1
    delta = 1
    old_delta = 0 
    
    if block in list:
        left_block = block
        right_block = block

    else:
        while delta != old_delta:
            trials += 1
            old_delta = delta   
            A = max(list[right] - block, 0)    # in standard binary search: A = 1
            B = max(block - list[left], 0)     # in standard binary search: B = 1
            middle = (A*left + B*right) // (A + B) 
            if list[middle] > block:
                right = middle
            elif list[middle] < block: 
                left = middle
            delta = right - left

        left_block = list[middle]
        right_block = list[min(middle+1, len(list)-1)]

    return(left_block, right_block, trials)


def true_cdf(block, max_block): 
    # Used in block_test
    # blocks uniformly distributed on {0, 1, ..., max_block}
    return((block + 1)/(max_block + 1))


def block_test(hash_blocks, n_nodes, base, block_size):

    # Approximated KS (Kolmogorov-Smirnov distance) between true random and PRNG
    # Computed only for blocks with block_size digits (each digit in base system)
    # More nodes means better approximation to KS

    hash_cdf = {}
    hash_blocks = OrderedDict(sorted(hash_blocks.items()))
    n_blocks = sum(hash_blocks.values())
    count = 0
    trials = 0  # total number of iterations in binary search


    for block in hash_blocks:
        hash_cdf[block] = count + hash_blocks[block]/n_blocks
        count = hash_cdf[block]

    cdf_list = list(hash_cdf.keys())
    max_block = base**block_size - 1
    KS = 0
    trials = 0     # total number of iterations in binary search
    arr_cdf = []   # theoretical cdf, values
    arr_ecdf = []  # empirical cdf (PRMG), values
    arr_arg = []   # arguments (block number) associated to cdf or ecdf

    for k in range(0, n_nodes): 

        block = int(0.5 + k * max_block / n_nodes)
        (left_block, right_block, trials) = get_closest_blocks(block, cdf_list, trials)
        cdf_val = true_cdf(block, max_block)
        ecdf_lval = hash_cdf[left_block]       # empirical cdf
        ecdf_rval = hash_cdf[right_block]      # empirical cdf
        ecdf_val = (ecdf_lval + ecdf_rval) / 2 # empirical cdf
        arr_cdf.append(cdf_val)
        arr_ecdf.append(ecdf_val)
        arr_arg.append(block)
        dist = abs(cdf_val - ecdf_val)
        if dist > KS:
            KS = dist

    return(KS, arr_cdf, arr_ecdf, arr_arg, trials) 


def autocorrel_test(cp_data, max_lag, base):

    mu = cp_data[0]
    cnt = cp_data[1]
    cp_vals = cp_data[2]
    cp_cnt = cp_data[3]

    mu /= cnt
    t_mu = (base-1) / 2
    var = cp_vals[0]/cp_cnt[0] - mu*mu
    t_var = (base*base -1) / 12 
    print()
    print("Digit mean: %6.2f (expected: %6.2f)" % (mu, t_mu))
    print("Digit var : %6.2f (expected: %6.2f)" % (var, t_var))
    print()
    print("Digit autocorrelations: ")
    for k in range(max_lag):
        autocorrel = (cp_vals[k]/cp_cnt[k] - mu*mu) / var
        print("Lag %4d: %7.4f" %(k, autocorrel))

    return()


#--- [3] Main section

# I tested (p, q) in {(3, 2), (7, 4), (13, 8), (401, 256)}

N = 64000             # number of digits to generate
p = 7                 # p/q must > 1, preferably >= 1.5 
q = 4                 # I tried q = 2, 4, 8, 16 and so on only 
base = q              # digit base, base <= q (base = 2 and base = q work) 
x0 = 50001            # seed to start the bigint sequence in PRNG
tau  = 41197          # co-seed of home-made PRNG
step = 37643          # co-seed of home-made PRNG
digit = -1            # fictitious digit before creating real ones 
block_size = 6        # digits per block for the block test; must be integer > 0
n_nodes = 500         # number of nodes for the block test
max_lag = 3           # for autocorrel test
seed = 104            # needed only with option = "Numpy"
np.random.seed(seed)  

#- [3.1] Home-made PRNG with parameters that work

p, q, base, block_size = 7, 4, 4, 6

# generate random digits, home-made PRNG
(hash_runs, hash_blocks, max_run, cp_data) = generate_digits(N, x0, p, q, tau, step,  
                                                             base, block_size, max_lag, 
                                                             option="Home-Made")
# run_test
run_test(base,max_run,hash_runs)

# block test
(KS, arr_cdf, arr_ecdf1, arr_arg1, trials) = block_test(hash_blocks, n_nodes, 
                                                        base ,block_size)
# autocorrel_test 
autocorrel_test(cp_data, max_lag, base) 

print()
print("Trials = ", trials)
print("KS = %8.5f\n\n" %(KS))


#- [3.2] Home-made, with parameters that don't work

p, q, base, block_size = 6, 4, 4, 6   

# generate random digits, home-made PRNG
(hash_runs, hash_blocks, max_run, cp_data) = generate_digits(N, x0, p, q, tau, step,  
                                                             base, block_size, max_lag, 
                                                             option="Home-Made")
# run_test
run_test(base,max_run,hash_runs)

# block test
(KS, arr_cdf, arr_ecdf2, arr_arg2, trials) = block_test(hash_blocks, n_nodes, 
                                                        base ,block_size)
# autocorrel_test 
autocorrel_test(cp_data, max_lag, base) 

print()
print("Trials = ", trials)
print("KS = %8.5f\n\n" %(KS))

#- [3.3] Home-made, another example of failure

p, q, base, block_size = 7, 4, 8, 6  

# generate random digits, home-made PRNG
(hash_runs, hash_blocks, max_run, cp_data) = generate_digits(N, x0, p, q, tau, step,  
                                                             base, block_size, max_lag, 
                                                             option="Home-Made")
# run_test
run_test(base,max_run,hash_runs)

# block test
(KS, arr_cdf, arr_ecdf3, arr_arg3, trials) = block_test(hash_blocks, n_nodes, 
                                                        base ,block_size)
# autocorrel_test 
autocorrel_test(cp_data, max_lag, base) 

print()
print("Trials = ", trials)
print("KS = %8.5f\n\n" %(KS))

#- [3.4] Home-made, using a large base

p, q, base, block_size = 401, 256, 256, 6  

# generate random digits, home-made PRNG
(hash_runs, hash_blocks, max_run, cp_data) = generate_digits(N, x0, p, q, tau, step,  
                                                             base, block_size, max_lag, 
                                                             option="Home-Made")
# run_test
run_test(base,max_run,hash_runs)

# block test
(KS, arr_cdf, arr_ecdf4, arr_arg4, trials) = block_test(hash_blocks, n_nodes, 
                                                        base ,block_size)
# autocorrel_test 
autocorrel_test(cp_data, max_lag, base) 

print()
print("Trials = ", trials)
print("KS = %8.5f\n\n" %(KS))


#- [3.5] Numpy PRNG (Mersenne twister)

# here p, q are irrelevant
base, block_size = 4, 6 

# generate random digits, home-made PRNG
(hash_runs, hash_blocks, max_run, cp_data) = generate_digits(N, x0, p, q, tau, step,  
                                                             base, block_size, max_lag, 
                                                             option="Numpy")
# run_test
run_test(base,max_run,hash_runs)

# block test
(KS, arr_cdf, arr_ecdf5, arr_arg5, trials) = block_test(hash_blocks, n_nodes, 
                                                        base ,block_size)
# autocorrel_test 
autocorrel_test(cp_data, max_lag, base) 

print()
print("Trials = ", trials)
print("KS = %8.5f\n\n" %(KS))


#--- [4] Scatterplot cdf (true random) versus ecdf (based on the two PRNGs) 

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

arr_cdf = np.array(arr_cdf) 
delta_ecdf1 = (np.array(arr_ecdf1) - arr_cdf) 
delta_ecdf2 = (np.array(arr_ecdf2) - arr_cdf) 
delta_ecdf3 = (np.array(arr_ecdf3) - arr_cdf) 
delta_ecdf4 = (np.array(arr_ecdf4) - arr_cdf) 
delta_ecdf5 = (np.array(arr_ecdf5) - arr_cdf) 

# print()
# print("blocks (arguments) used to compute ecdf1:\n")
# print(arr_arg1)

plt.plot(delta_ecdf1, linewidth = 0.4, color = 'red', alpha = 1)
plt.plot(delta_ecdf2, linewidth = 0.3, color = 'blue', alpha = 0.5)
plt.plot(delta_ecdf3, linewidth = 0.3, color = 'darkorange', alpha = 1)
plt.plot(delta_ecdf4, linewidth = 0.4, color = 'purple', alpha = 1)
plt.plot(delta_ecdf5, linewidth = 0.4, color = 'darkgreen', alpha = 1)
plt.axhline(y = 0.0, linewidth = 0.5, color = 'black', linestyle = 'dashed')
plt.legend(['HM1: good', 'HM2: bad', 'HM3: bad', 'HM4: good', 'Numpy'], 
             loc='lower right', prop={'size': 7}, )
plt.show()
