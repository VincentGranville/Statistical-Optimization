import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl


#--- [1] Read data

data = pd.read_csv("spx500.txt") 

def date_to_int(date, start_date):
    xstart_date = datetime.strptime(start_date, '%b %d %Y')
    xdate = datetime.strptime(date, '%b %d %Y')
    date_int = str(xdate - xstart_date)
    date_int = date_int.split(' ')[0]
    if date_int == '0:00:00':
        date_int = 0
    else:
        date_int = int(date_int)
    return(date_int)

arr_date  = data['Date']
arr_low   = data['Low']
arr_high  = data['High']
arr_open  = data['Open']
arr_idate = []
arr_tdate  = []
nobs = len(arr_date)

for k in range(nobs):
    arr_idate.append(date_to_int(arr_date[k], arr_date[0]))
    tdate = datetime.strptime(arr_date[k], '%b %d %Y')
    arr_tdate.append(tdate)

data.insert(1, 'iDate', arr_idate, True)

print(data.head())
print(data.columns.values)
print(data.shape)
# features: 'Date','iDate','Open','High','Low','Close','Adj_Close','Volume'

#--- [2] Initializations for main loop

h_params = {}

l_init = (4000, 5000, 6000, 7000)
l_duration = (1000, 2000, 3000)
l_windowLow  = (40, 60) 
l_buy_param  = (1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5) 
l_sell_param = (1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5) 

h_max = {}
h_min = {}

#- [2.1] preprocessing to speed up computations

for k in range(nobs):
    for windowLow in l_windowLow:
        if k >= windowLow: 
            h_min[(k, windowLow)] = min(arr_low[k-windowLow:k-1])

#- [2.2] create table of parameter sets (stored as key in h_params)

for buy_param in l_buy_param:
    for sell_param in l_sell_param:
        for windowLow in l_windowLow: 
            for init in l_init:
                for duration in l_duration: 
                    end = init + duration
                    key = (init,end,buy_param,sell_param,windowLow)
                    h_params[key] = 1

#- [2.3] Create main hash tables 
#        The index (same in all tables) is the parameter set

bottom      = {} 
hold        = {} # true if we have open position
value       = {}
max_hold    = {}
nbuys       = {} # number buys done during whole period
entry_price = {} # buy price on first buy 
sell_price  = {} # sell price on last sell, before current buy
wins        = {} # a win is buying a lower price than last sell

init_price  = {} # price when trading period starts
buy_price   = {} # buy price
end_price   = {} # price when trading period ends
entry_idate = {} # date of first buy
exit_idate  = {} # date of last sell if out, otherwise end_idate
init_idate  = {} # date when trading period starts
end_idate   = {} # date when trading period ends
start       = {}

for params in h_params:

    bottom[params]      = arr_high[0]
    hold[params]        = False 
    value[params]       = 0
    max_hold[params]    = 0
    nbuys[params]       = 0
    entry_price[params] = -1
    sell_price[params]  = -1
    wins[params]        = 0  


#--- [3] Main loop
 
for k in range(nobs):
    # loop over all trading days

    idate = arr_idate[k]
    price_high = arr_high[k]
    price_low  = arr_low[k]
    if k % 100 == 0:
        print(arr_date[k])

    for params in h_params:
        # update trading stats for each param set in parallel

        init       = params[0]
        end        = params[1]
        buy_param  = params[2]
        sell_param = params[3] 
        windowLow  = params[4] 
        rnd        = 0.5

        price = rnd * price_high + (1-rnd) * price_low 
  
        if k == init:
            init_price[params] = price
            init_idate[params] = idate
        elif k == end:  
            end_price[params] = price
            end_idate[params] = idate

        if k >= windowLow and price < h_min[(k, windowLow)]:
            bottom[params] = price

        if k >= init and not hold[params] and k < end:
            if price < buy_param * bottom[params]: 
                # buy
                buy_price[params] = price
                if value[params] == 0:
                    # first purchase
                    value[params] = price
                    entry_price[params] = price
                    entry_idate[params] = idate
                    if price < init_price[params]: 
                        # first buy at lower price than init price
                        wins[params] += 1
                elif buy_price[params] < sell_price[params]:
                    wins[params] += 1
                start[params] = idate
                hold[params] = True
                nbuys[params] += 1

        elif hold[params] and k < end:
            span = idate-start[params]
            if span > max_hold[params]:
                max_hold[params] = span
            if price > sell_param * buy_price[params]:  
                # sell
                sell_price[params] = price
                value[params] *= price/buy_price[params]
                hold[params]= False
                exit_idate[params] = idate


#--- [4] Summary stats for each param set

# group params in h_params by (init, end)
# stored average stats by grouped params, in arr_local

hash_performance = {}
hash_count = {}

for params in h_params:
    if hold[params]:
        value[params] *= end_price[params]/buy_price[params]
        exit_idate[params] = end_idate[params]
    else:
        if end_price[params] < sell_price[params]:
            wins[params] += 1
    duration1 = end_idate[params] - init_idate[params] + 1
    duration2 = exit_idate[params] - entry_idate[params] + 1
    R_market = end_price[params] / init_price[params]
    R_strategy = value[params] / entry_price[params]
    adj_R_market = 100*(R_market**(365/duration1) - 1)
    adj_R_strategy = 100*(R_strategy**(365/duration1) - 1)
    ratio = R_strategy / R_market  # reinvest all to compound return
    performance = adj_R_strategy - adj_R_market
    if performance > 0:
        success = 1
    else: 
        success = 0

    key = (params[2], params[3], params[4])
    arr_local = [nbuys[params], wins[params], 
                 performance, adj_R_market, adj_R_strategy,
                 duration1, duration2, success, hold[params]]  

    if key in hash_performance:
        arr_local2 = hash_performance[key]
        for idx in range(len(arr_local2)):
            arr_local2[idx] += arr_local[idx]
        hash_performance[key] = arr_local2 
        hash_count[key] += 1
    else:
        hash_performance[key] = arr_local 
        hash_count[key] = 1

#- [4.1] to get averages for each param set, divide sums by sample size cnt 

cnt = hash_count[key]

for key in hash_performance:
    arr_avg = hash_performance[key]
    for idx in range(len(arr_avg)):
        arr_avg[idx] /= cnt


#--- [5] Print results

OUT = open("spx500-results2.txt", "w")

labels="beta\tsigma\tomega\tsample size\tbuys\tbuys below last sell\tDelta return"	
labels+="base return\ttrading return\tperiod (days)\tactive peiod\tsuccess rate\thold rate"
OUT.write(labels + "\n")

for key in hash_count:  
    strout = "" 
    for param in key:
        strout += str(param) + "\t"
    cnt = hash_count[key]
    strout += str(cnt) + "\t"
    arr_avg = hash_performance[key]
    for idx in range(len(arr_avg)):
        strout += str(arr_avg[idx]) + "\t"
    strout += "\n"
    OUT.write(strout)

OUT.close()


#--- [6] Visualizations

hash_xy = {}
hash_color = {}
for value in l_windowLow:
    hash_xy[value] = []
    hash_color[value] = []

for key in hash_count:
    cnt = hash_count[key]
    arr_avg = hash_performance[key]
    for params in key:
        buy_param  = key[0]
        sell_param = key[1]
        window_low = key[2]
        performance = arr_avg[2]  # performance 
        if performance > 0.5:
            color = [0, 0, 1]     # blue
        elif performance > 0.1:
            color = [0, 1, 0]     # lightgreen
        elif performance > -0.1:
            color = [0.6, 0.6, 0.6] # lightgray
        elif performance > -0.5:
            color = [1, 0.6, 0] # orange
        else:
            color = [1, 0, 0]  # red
        local_arr = hash_xy[window_low] 
        local_arr.append([buy_param, sell_param])
        local_arr = hash_color[window_low] 
        local_arr.append(color)

mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

for window_low in hash_xy:
    z = np.array(hash_xy[window_low])
    z = np.transpose(z)
    color = hash_color[window_low]
    plt.scatter(z[0], z[1], c = color)
    plt.show()
        
