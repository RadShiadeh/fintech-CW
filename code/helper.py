import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from BSE import market_session
""" this code is to help run the bse simulations on the jupyter notebook to avoid the notebook being too long"""

def make_df(path: str):
    #makes data frames from the avg balance csv file
    df = pd.read_csv(path)
    df.columns =  ['name', 'time', 'curr best bid', 'curr best offer', 'trader1', 'total profit1', 'no. 1', 'avg profit1', 'trader2', 'total profit2', 'no. 2', 'avg profit2', 'err']
    df_profit = df[['avg profit1', 'avg profit2']]
    df_profit.columns = ['SHVR', 'ZIC']
    return df, df_profit

def collect_avg_profit(df):
    #returns average total profit of each trader type given a data frame
    _zic = df['ZIC'][len(df)-1]
    _shvr = df['SHVR'][len(df)-1]
    return _zic, _shvr

def plot_performance(n50mean_zic, n500mean_zic, n50mean_shvr, n500mean_shvr):
    #plot average profit of zic and shvr w different independant sessions over time period
    fig = plt.figure(figsize=(20, 7.5))
    ax1 = fig.add_subplot(221)
    ax1.plot(n50mean_zic, 'b', n50mean_shvr, 'r')
    ax2 = fig.add_subplot(222)
    ax2.plot(n500mean_zic, 'b', n500mean_shvr, 'r')

    ax1.title.set_text('zic vs shvr 50 sessions')
    ax2.title.set_text('zic vs shvr 500 sessions')

def compare(shvr, zic):
    #compare and collect the number of wins for shvr and zic traders given their average profit arrays
    shvr_w = 0
    zic_w = 0
    for i in range(len(shvr)):
        if shvr[i] - zic[i] > 0:
            shvr_w += 1
        else:
            zic_w += 1
    
    return shvr_w, zic_w

def trader_specs_two(R, n):
    #defines the trader specs
    SHVR_num = (n*R)//100
    zic_num = ((100-R)*n)//100
    buyer_spec = [('SHVR', SHVR_num), ('ZIC', zic_num)]
    seller_spec = [('SHVR', SHVR_num), ('ZIC', zic_num)]
    trader_specs = {'sellers': seller_spec, 'buyers': buyer_spec}
    return trader_specs

def run_market_sim(trial_id, no_sessions, R, n, supply_range, demand_range, start_time, end_time, path):
    #simulate the market session and collect the average profit of each trader at the end of each independant session
    trader_specs = trader_specs_two(R, n)
    total_avg_zic = []
    total_avg_shvr = []
    
    for _ in range(no_sessions):
        supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [supply_range], 'stepmode': 'fixed'}]
        demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [demand_range], 'stepmode': 'fixed'}]
        order_interval = 60
        order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                    'interval': order_interval, 'timemode': 'periodic'}
        dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                            'dump_avgbals': True, 'dump_tape': False}

        verbose = False
        market_session(trial_id, start_time, end_time, trader_specs, order_sched, dump_flags, verbose)
        _, df_profit = make_df(path)
        _zic, _shvr = collect_avg_profit(df_profit)
        total_avg_shvr.append(_shvr)
        total_avg_zic.append(_zic)
    
    return total_avg_shvr, total_avg_zic

def R_market_run(R, no_sessions, n, supply_range, demand_range, start_time, end_time):
    #simulate market sessions for different ratio values R which defines the zic and shvr numbers
    res = []
    tmp = "n" + str(no_sessions) + "_"
    for r in R:
        trial_id = tmp + str(r)
        path = str(trial_id) + "_avg_balance.csv"
        res.append(run_market_sim(trial_id, no_sessions, r, n, supply_range, demand_range, start_time, end_time, path))

    return res

def plot_wins(res50: list, res500: list):
    #plot, for different ratios of zic and shvr traders, how many times they made more profit than the other in their independant sessions
    shvr_win50 = [0] * 9
    zic_win50 = [0] * 9
    zic_win500 = [0] * 9
    shvr_win500 = [0] * 9
    for i in range(9):
        shvr50 = res50[i][0]
        zic50 = res50[i][1]
        shvr500 = res500[i][0]
        zic500 = res500[i][1]
        for z in range(len(zic500)):
            if shvr500[z] > zic500[z]:
                shvr_win500[i] += 1
                zic_win500[i] = len(zic500) - shvr_win500[i]            
        for j in range(len(zic50)):
            if shvr50[j] > zic50[j]:
                shvr_win50[i] += 1
                zic_win50[i] = len(zic50) - shvr_win50[i]

    fig = plt.figure(figsize=(20, 7.5))
    ax1 = fig.add_subplot(221)
    ax1.plot(shvr_win50, 'b', label='shvr')
    ax1.plot(zic_win50, 'r', label='zic')
    ax1.legend()
    ax2 = fig.add_subplot(222)
    ax2.plot(shvr_win500, 'b', label='shvr')
    ax2.plot(zic_win500, 'r', label='zic')
    ax2.legend()

def collect_pvals_norm(marketoutput: list):
    #test normality of data collected in simulated market sessions
    res = []
    for i in range(len(marketoutput)):
        _, p_shvr = stats.shapiro(marketoutput[i][0])
        _, p_zic = stats.shapiro(marketoutput[i][1])
        p_res_50 = [p_shvr, p_zic]
        res.append(p_res_50)
    return res

def A_B_test(p_val: list, data_: list, a: float = 0.05):
    #A/B testing on two data, use either mann-whitney U test or t-test depending on normality of the data (when there are different ratios of zic and shvr)
    res = []

    for i in range(len(p_val)):
        if p_val[i][0] < a or p_val[i][1] < a:
            _, pval = stats.mannwhitneyu(data_[i][0], data_[i][1])
            print("n="+ str(len(data_[0][0])) + ", ratio: "+ str((i+1))+ " to "+ str(9-i) + " is NOT normal used NON-parametric test, p val is:", "{:.4f}".format(pval))
            res.append(pval)
        elif p_val[i][0] > a and p_val[i][1] > a:
            _, pval = stats.ttest_ind(data_[i][0], data_[i][1], equal_var=False)
            print("n="+ str(len(data_[0][0])) + ", ratio: "+ str((i+1))+ " to "+ str(9-i)+" is NORMAL used parametric test, p val is:", "{:.4f}".format(pval))
            res.append(pval)
    
    return res

def a_b_c_test(shvr_avg, gvwy_avg, zic_avg, zip_avg, res):
    #use either kruksal or one way ANOVA test depending on the normality of the data
    if res[0][0] < 0.05 or res[0][1] < 0.05 or res[0][2] < 0.05 or res[0][3] < 0.05:
        _, p = stats.kruskal(shvr_avg, gvwy_avg, zic_avg, zip_avg)
    else:
        _, p = stats.f_oneway(shvr_avg, gvwy_avg, zic_avg, zip_avg)
    return p

def collect_mean_4(df):
    #returns accum average profit of each trader type 
    mean_zic = df['ZIC'][len(df)-1]
    mean_shvr = df['SHVR'][len(df)-1]
    mean_GVWY = df['GVWY'][len(df)-1]
    mean_zip = df['ZIP'][len(df)-1]
    return mean_shvr, mean_GVWY, mean_zic, mean_zip

def df_four(path):
    #make dataframe from the simulated market sessions when there are 4 different traders
    df = pd.read_csv(path)
    df.columns =  ['name', 'time', 'curr best bid', 'curr best offer', 'trader1', 'total profit1', 'no. 1', 'avg profit1', 'trader2', 'total profit2', 'no. 2', 'avg profit2',
                   'trader3', 'total profit3', 'no. 3', 'avg profit3', 'trader4', 'total profit4', 'no. 4', 'avg profit4', 'err']
    df_profit = df[['avg profit1', 'avg profit2', 'avg profit3', 'avg profit4']]
    df_profit.columns = ['GVWY', 'SHVR', 'ZIC', 'ZIP']
    return df, df_profit    

def run_market_sim_four(trial_id, no_sessions, t, n, supply_range, demand_range, start_time, end_time):
    #simulate a market sim for 4 traders and collect their profit as well as if the data is normal or not (p)
    res = [[] for _ in range(4)]

    seller_spec = [('SHVR', int(t[0]*n/100)), ('GVWY', int(t[1]*n/100)), ('ZIC', int(t[2]*n/100)), ('ZIP', int(t[3]*n/100))]
    buyer_spec = [('SHVR', int(t[0]*n/100)), ('GVWY', int(t[1]*n/100)), ('ZIC', int(t[2]*n/100)), ('ZIP', int(t[3]*n/100))]
    trader_specs = {'sellers': seller_spec, 'buyers': buyer_spec}
    path = str(trial_id) + "_avg_balance.csv"
    for _ in range(no_sessions):
        supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [supply_range], 'stepmode': 'fixed'}]
        demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [demand_range], 'stepmode': 'fixed'}]
        order_interval = 60
        order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                            'interval': order_interval, 'timemode': 'periodic'}
        dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                            'dump_avgbals': True, 'dump_tape': False}
        verbose = False
        market_session(trial_id, start_time, end_time, trader_specs, order_sched, dump_flags, verbose)
        _, df_profit = df_four(path)
        mean_shvr, mean_GVWY, mean_zic, mean_zip = collect_mean_4(df_profit)
        res[0].append(mean_shvr)
        res[1].append(mean_GVWY)
        res[2].append(mean_zic)
        res[3].append(mean_zip)
    
    p = if_norm(res)
    
    result = a_b_c_test(res[0], res[1], res[2], res[3], p)
    

            
    return res, result

def if_norm(res):
    # check if all vals are normal or not using shapiro-wilk
    p_vals = []
    _, p_shvr = stats.shapiro(res[0])
    _, p_gvwy = stats.shapiro(res[1])
    _, p_zic = stats.shapiro(res[2])
    _, p_zip = stats.shapiro(res[3])
    p_vals.append([p_shvr, p_gvwy, p_zic, p_zip])

    return p_vals

def plot_performance_same_ratio(res50):
    # for when there are equal number of trader of each type, plot their performance
    shvr_w = 0
    gvwy_w = 0
    zic_w = 0
    zip_w = 0
    for i in range(len(res50[0])):
        if res50[0][i] > res50[1][i] and res50[0][i] > res50[2][i] and res50[0][i] > res50[3][i]:
            shvr_w += 1
        elif res50[1][i] > res50[0][i] and res50[1][i] > res50[2][i] and res50[1][i] > res50[3][i]:
            gvwy_w += 1
        elif res50[2][i] > res50[0][i] and res50[2][i] > res50[1][i] and res50[2][i] > res50[3][i]:
            zic_w += 1
        else:
            zip_w+=1
    
    xlabels = ['shvr', 'gvwy', 'zic', 'zip']
    vals = [shvr_w, gvwy_w, zic_w, zip_w]
    col = ['tab:red', 'tab:green', 'tab:blue', 'tab:red']
    fig = plt.figure(figsize=(20, 7.5))
    ax1 = fig.add_subplot(221)
    ax1.plot(res50[0], 'r', label='shvr')
    ax1.plot(res50[1], 'g', label='gvwy')
    ax1.plot(res50[2], 'b', label='zic')
    ax1.plot(res50[3], 'y', label='zip') #res50[1], 'g', res50[2], 'b', res50[3], 'y'
    ax1.set_xlabel('session')
    ax1.set_ylabel('average profit')
    ax1.title.set_text('zic vs zip vs shvr vs gvwy avg profit')
    ax1.legend()
    ax2 = fig.add_subplot(222)
    ax2.bar(xlabels, vals, color=col)
    ax2.set_xlabel("trader")
    ax2.set_ylabel("wins")
    ax2.title.set_text('same ratio, number of wins')

def gather_wins(res):
    #gather and return the number of times a trader had more profit than the rest in a given session
    gvwy_wins = [0] * 4
    shvr_wins = [0] * 4
    zic_wins = [0] * 4
    zip_wins = [0] * 4

    for i in range(4):
        shvr = res[i][0]
        gvwy = res[i][1]
        zic = res[i][2]
        zip_ = res[i][3]
        for z in range(len(shvr)):
            if shvr[z] > gvwy[z] and shvr[z] > zic[z] and shvr[z] > zip_[z]:
                shvr_wins[i] += 1
            if gvwy[z] > shvr[z] and gvwy[z] > zic[z] and gvwy[z] > zip_[z]:
                gvwy_wins[i] += 1
            if zic[z] > shvr[z] and zic[z] > gvwy[z] and zic[z] > zip_[z]:
                zic_wins[i] += 1
            if zip_[z] > shvr[z] and zip_[z] > zic[z] and zip_[z] > gvwy[z]:
                zip_wins[i] += 1
    return shvr_wins, gvwy_wins, zic_wins, zip_wins

def sub_plot_add(shvrw, gvwyw, zicw, zipw, fig, index, ratio, axID, n):
    #helper for plot_wins_4 to make the sub plots
    ax = "ax"+str(axID)
    ax = fig.add_subplot(index)
    ax.plot(shvrw, 'r', label='shvr')
    ax.plot(gvwyw, 'g', label='gvwy')
    ax.plot(zicw, 'b', label='zic')
    ax.plot(zipw, 'y', label='zip')
    title = 'for ratio ' + str(ratio) +', trader wins per permutation in ' + str(n) + ' sessions'
    ax.title.set_text(title)
    ax.legend()


def plot_wins_4(res1: list, ratio1: list, res2: list, ratio2: list, res3: list, ratio3: list):
    #plot the number of time a given trader in a session has made more profit than the rest
    shvr_wins1, gvwy_wins1, zic_wins1, zip_wins1 = gather_wins(res1)
    shvr_wins2, gvwy_wins2, zic_wins2, zip_wins2 = gather_wins(res2)
    shvr_wins3, gvwy_wins3, zic_wins3, zip_wins3 = gather_wins(res3)

    fig1 = plt.figure(figsize=(20, 7))
    
    sub_plot_add(shvr_wins1, gvwy_wins1, zic_wins1, zip_wins1, fig1, 221, ratio1, 1, len(res1[0][0]))
    sub_plot_add(shvr_wins2, gvwy_wins2, zic_wins2, zip_wins2, fig1, 222, ratio2, 2, len(res2[0][0]))
    sub_plot_add(shvr_wins3, gvwy_wins3, zic_wins3, zip_wins3, fig1, 223, ratio3, 3, len(res3[0][0]))



def run_market_sim_D(trial_id, no_sessions, supply_range, demand_range, start_time, end_time, path, path_strat, buyer_spec, seller_spec):
    # simulate a market session and collect profit per sec, hyperparams and profit of zipsh and zic traders to replicate cliff 2023 part D 1
    trader_specs = {'sellers': seller_spec, 'buyers': buyer_spec}
    total_avg_zipsh = []
    avg_pps_total = []
    total_avg_prof_per_session = []
    hyper_params = [[] for _ in range(5)]
    
    for _ in range(no_sessions):
        supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [supply_range], 'stepmode': 'fixed'}]
        demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [demand_range], 'stepmode': 'fixed'}]
        order_interval = 30
        order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                    'interval': order_interval, 'timemode': 'periodic'}
        dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': True,
                            'dump_avgbals': True, 'dump_tape': False}

        verbose = False
        market_session(trial_id, start_time, end_time, trader_specs, order_sched, dump_flags, verbose)
        df_strats, df_profit = make_df_D(path, path_strat)
        b, m, ca, cr, mb = collect_hyperparams(df_strats)
        hyper_params[0].append(b)
        hyper_params[1].append(m)
        hyper_params[2].append(ca)
        hyper_params[3].append(cr)
        hyper_params[4].append(mb)        
        _, _zipsh, avg_pps, avg_prof_per_sec = collect_avg_profit_D(df_profit)
        total_avg_zipsh.append(_zipsh)
        avg_pps_total.append(avg_pps)
        total_avg_prof_per_session.append(avg_prof_per_sec)
    
    return total_avg_zipsh, avg_pps_total, total_avg_prof_per_session, hyper_params

def run_market_d2(zipsh_num, k, zic_num, range1, range2, start_time, mid_time, end_time, ot, id, n, path, path_strat):
    # simulate a market session and collect all results for when zipsh is in the market, return vals are average prof per independant session and all hyperparameters
    total_avg_prof_per_session = []
    hyper_params = [[] for _ in range(5)]
    buyers_spec = [('ZIPSH', zipsh_num, {'k': k, 'optimizer': 'ZIPSH'}), ('ZIC', zic_num)]
    sellers_spec = [('ZIC', zic_num)]
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

    for _ in range(n):
        supply_schedule = [{'from': start_time, 'to': mid_time, 'ranges': [range1], 'stepmode': 'fixed'},
                        {'from': mid_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]
        demand_schedule = supply_schedule

        order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                    'interval': ot, 'timemode': 'periodic'}
        trial_id = id
        dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': True,
                    'dump_avgbals': True, 'dump_tape': False}
        verbose = False
        market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)
        df_strats, df_profit = make_df_D(path, path_strat)
        b, m, ca, cr, mb = collect_hyperparams(df_strats)
        hyper_params[0].append(b)
        hyper_params[1].append(m)
        hyper_params[2].append(ca)
        hyper_params[3].append(cr)
        hyper_params[4].append(mb)        
        _, _, _, avg_prof_per_sec = collect_avg_profit_D(df_profit)
        total_avg_prof_per_session.append(avg_prof_per_sec)
        
    return total_avg_prof_per_session, hyper_params

def make_df_D(path: str, path_strat):
    #make dataframes from the csv files average and strats 
    df = pd.read_csv(path)
    df.columns =  ['name', 'time', 'curr best bid', 'curr best offer', 'trader1', 'total profit1', 'no. 1', 'avg profit1', 'trader2', 'total profit2', 'no. 2', 'avg profit2', 'err']
    df_profit = df[['time', 'avg profit1', 'avg profit2']]
    df_profit.columns = ['time', 'ZIC', 'ZIPSH']

    imap = {1: 'time', 26: 'mBuy', 30: 'beta', 32: 'momentum', 34: 'c_r', 36: 'c_a'}

    df_st = pd.read_csv(path_strat)
    cols = []
    for i in range(38):
        if i in imap:
            cols.append(imap[i])
        else: 
            cols.append('idc')
    df_st.columns = cols
    df_strat = df_st[['time', 'mBuy', 'beta', 'momentum', 'c_r', 'c_a']]

    return df_strat, df_profit

def collect_hyperparams(df):
    #use dataframe from created from strats to extract hyperparameters
    beta = []
    momentum = []
    c_r = []
    c_a = []
    mBuy = []
    for i in range(len(df)):
        beta.append(df['beta'][i])
        momentum.append(df['momentum'][i])
        c_r.append(df['c_r'][i])
        c_a.append(df['c_a'][i])
        mBuy.append(df['mBuy'][i])
    
    return beta, momentum, c_a, c_r, mBuy

def collect_avg_profit_D(df):
    #use created datafram to collect the total profit of each trader as well as average profit per second for the zipsh trader
    average_pps_per_day = []
    prof_per_sec = []
    _zic = df['ZIC'][len(df)-1]
    _zipsh = df['ZIPSH'][len(df)-1]
    _avg_pps = _zipsh/df['time'][len(df)-1]
    n = 1
    for i in range(len(df)):
        if i < 1:
            continue

        if df['time'][i] == df['time'][i-1]:
            continue

        if df['ZIPSH'][i] < df['ZIPSH'][i-1] and len(prof_per_sec) != 0:
            val = (sum(prof_per_sec)/len(prof_per_sec))
            if val <= 0:
                val = 0
            average_pps_per_day.append(val)
            prof_per_sec = []
            continue

        pf = (df['ZIPSH'][i] - df['ZIPSH'][i-1])/(df['time'][i] - df['time'][i-1])
        if pf < 0:
            pf = 0
        prof_per_sec.append(pf)

        if df['time'][i] >= 60*60*24*n and len(prof_per_sec) != 0:
            n+=1                
            val = (sum(prof_per_sec)/len(prof_per_sec))
            if val <= 0:
                val = 0
            average_pps_per_day.append(val)
            prof_per_sec = []

    return _zic, _zipsh, _avg_pps, average_pps_per_day

def plot_params_pps(tpi, betas, mom, c_a, c_r, mBuy, n):
    #tpi = total_avg_profit_perS[i]
    #plot all hyper parameters and average profit
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(221)
    ax1.plot(tpi, 'r', label='zipsh') #res50[1], 'g', res50[2], 'b', res50[3], 'y'
    ax1.set_xlabel('changes')
    ax1.set_ylabel('average pps')
    ax1.title.set_text('avg pps for zipsh in the session ' + str(n))
    ax1.legend()
    ax2 = fig.add_subplot(222)
    ax2.plot(betas, 'b', label='beta')
    ax2.set_xlabel('changes')
    ax2.set_ylabel('vals')
    ax2.title.set_text('beta vals in session: ' + str(n))
    ax2.legend()
    ax3 = fig.add_subplot(223)
    ax3.plot(mom, 'b', label='momentum')
    ax3.set_xlabel('changes')
    ax3.set_ylabel('vals')
    ax3.title.set_text('momentum vals in session: ' + str(n))
    ax3.legend()
    ax4 = fig.add_subplot(224)
    ax4.plot(mBuy, 'b', label='margin buy')
    ax4.set_xlabel('changes')
    ax4.set_ylabel('vals')
    ax4.title.set_text('margin buy vals in session: ' + str(n))
    ax4.legend()
    fig2 = plt.figure(figsize=(20, 7.5))
    ax = fig2.add_subplot(221)
    ax.plot(c_a, 'b', label='c_a')
    ax.set_xlabel('changes')
    ax.set_ylabel('vals')
    ax.title.set_text('c_a vals in session: ' + str(n))
    ax.legend()
    ax5 = fig2.add_subplot(222)
    ax5.plot(c_r, 'b', label='c_r')
    ax5.set_xlabel('changes')
    ax5.set_ylabel('vals')
    ax5.title.set_text('c_r vals in session: ' + str(n))
    ax5.legend()


def test(taps):
    #collect averages at the end of each permutation, check for normality and then either use parametric t-test or non-parametric wilcoxen test for part D 1
    avg_total = []
    for i in range(len(taps)):
        avg_end_eachD = []
        for j in range(len(taps[i])-1):
            if taps[i][j] == 0:
                continue
            pf = (taps[i][j+1] *100) / taps[i][j]
            if pf <=0:
                avg_end_eachD.append(0)
            else:
                avg_end_eachD.append(pf)
        avg_total.append(np.mean(avg_end_eachD))

    _, p = stats.shapiro(avg_total)

    if p < 0.05:
        print("not normal, using wilcoxen")
        _, f = stats.wilcoxon(avg_total)
        return f
    else:
        print("normal, using t-test")
        _, f = stats.ttest_1samp(avg_total, 0)
        return f

def test_2(avg_prof):
    #collect averages at the end of each permutation, check for normality and then either use parametric t-test or non-parametric wilcoxen test for part D 2
    avg_per_sess = []
    for i in range(len(avg_prof)):
        sess = avg_prof[i]
        avg_per_sess.append(np.mean(sess))
    _, p = stats.shapiro(avg_per_sess)

    if p < 0.05:
        print("not normal, using wilcoxen")
        _, pval = stats.wilcoxon(avg_per_sess)
        return pval
    else:
        print("normal, using t-test")
        _, pval = stats.ttest_1samp(avg_per_sess, 0)
        return pval