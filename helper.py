# Some code...
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import mstats
from BSE import market_session

def make_df(path: str):
    df = pd.read_csv(path)
    df.columns =  ['name', 'time', 'curr best bid', 'curr best offer', 'trader1', 'total profit1', 'no. 1', 'avg profit1', 'trader2', 'total profit2', 'no. 2', 'avg profit2', 'err']
    df_profit = df[['avg profit1', 'avg profit2']]
    df_profit.columns = ['SHVR', 'ZIC']
    return df, df_profit

def collect_avg_profit(df):
    _zic = df['ZIC'][len(df)-1]
    _shvr = df['SHVR'][len(df)-1]
    return _zic, _shvr

def plot_performance(n50mean_zic, n500mean_zic, n50mean_shvr, n500mean_shvr):
    plt.figure(figsize=(10, 4))
    plt.plot(n50mean_zic, 'r', n50mean_shvr, 'b')
    plt.title('50 sessions')
    plt.xlabel('session')
    plt.ylabel('profit')
    plt.show()
    plt.figure(figsize=(20, 4))
    plt.plot(n500mean_zic, 'r', n500mean_shvr, 'b')
    plt.title('500 sessions')
    plt.xlabel('session')
    plt.ylabel('profit')

def trader_specs_two(R, n):
    SHVR_num = (n*R)//100
    zic_num = ((100-R)*n)//100
    buyer_spec = [('SHVR', SHVR_num), ('ZIC', zic_num)]
    seller_spec = [('SHVR', SHVR_num), ('ZIC', zic_num)]
    trader_specs = {'sellers': seller_spec, 'buyers': buyer_spec}
    return trader_specs

def run_market_sim(trial_id, no_sessions, R, n, supply_range, demand_range, start_time, end_time, path):
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
    res = []
    tmp = "n" + str(no_sessions) + "_"
    for r in R:
        trial_id = tmp + str(r)
        path = str(trial_id) + "_avg_balance.csv"
        res.append(run_market_sim(trial_id, no_sessions, r, n, supply_range, demand_range, start_time, end_time, path))

    return res

def plot_wins(res: list):
    shvr_win = [0] * 9
    zic_win = [0] * 9
    for i in range(9):
        shvr = res[i][0]
        zic = res[i][1]
        for j in range(len(zic)):
            if shvr[j] > zic[j]:
                shvr_win[i] += 1
                zic_win[i] = len(zic) - shvr_win[i]
    plt.figure(figsize=(10, 6))
    plt.plot(shvr_win, 'r', zic_win, 'b')
    plt.title('zic vs shvr wins for ' + str(len(zic)) + ' sessions')
    plt.xlabel('ratios')
    plt.ylabel('number of wins')
    plt.show()


def collect_pvals_norm(marketoutput: list):
    res = []
    for i in range(len(marketoutput)):
        _, p_shvr = stats.shapiro(marketoutput[i][0])
        _, p_zic = stats.shapiro(marketoutput[i][1])
        p_res_50 = [p_shvr, p_zic]
        res.append(p_res_50)
    return res

def A_B_test(p_val: list, data_: list, a: float = 0.05):
    res = []

    for i in range(len(p_val)):
        if p_val[i][0] < a or p_val[i][1] < a:
            _, pval = stats.mannwhitneyu(data_[i][0], data_[i][1])
            print("n="+ str(len(data_[0][0])) + " ratio: "+ str((i+1))+ " to "+ str(9-i) + " is NOT normal used NON-parametric test, p val is:", pval)
            res.append(pval)
        elif p_val[i][0] > a and p_val[i][1] > a:
            _, pval = stats.ttest_ind(data_[i][0], data_[i][1], equal_var=False)
            print("n="+ str(len(data_[0][0])) + " ratio: "+ str((i+1))+ " to "+ str(9-i)+" is NORMAL used parametric test, p val is:", pval)
            res.append(pval)
    
    return res

def anova_test(df):
    shvr_avg = df['SHVR']
    gvwy_avg = df['GVWY']
    zic_avg = df['ZIC']
    zip_avg = df['ZIP']
    _, p = stats.f_oneway(shvr_avg, gvwy_avg, zic_avg, zip_avg)
    return p

def collect_mean_4(df):
    mean_zic = df['ZIC'][len(df)-1]
    mean_shvr = df['SHVR'][len(df)-1]
    mean_GVWY = df['GVWY'][len(df)-1]
    mean_zip = df['ZIP'][len(df)-1]
    return mean_shvr, mean_GVWY, mean_zic, mean_zip

def df_four(path):
    df = pd.read_csv(path)
    df.columns =  ['name', 'time', 'curr best bid', 'curr best offer', 'trader1', 'total profit1', 'no. 1', 'avg profit1', 'trader2', 'total profit2', 'no. 2', 'avg profit2',
                   'trader3', 'total profit3', 'no. 3', 'avg profit3', 'trader4', 'total profit4', 'no. 4', 'avg profit4', 'err']
    df_profit = df[['avg profit1', 'avg profit2', 'avg profit3', 'avg profit4']]
    df_profit.columns = ['GVWY', 'SHVR', 'ZIC', 'ZIP']
    return df, df_profit    

def four_trader_specs(t: list, n: int):
    res = []
    for i in range(len(t)):
        t = np.roll(t, i)
        tmp = []
        for x in t:
            tmp.append(int(x))
        t = tmp
        shvr_num = (t[0]*n)/100
        gvwy_num = (t[1]*n)/100
        zic_num = (t[2]*n)/100
        zip_num = (t[3]*n)/100
        buyer_specs = [('SHVR', int(shvr_num)), ('GVWY', int(gvwy_num)), ('ZIC', int(zic_num)), ('ZIP', int(zip_num))]
        seller_specs = [('SHVR', int(shvr_num)), ('GVWY', int(gvwy_num)), ('ZIC', int(zic_num)), ('ZIP', int(zip_num))]
        trader_specs = {'sellers': seller_specs, 'buyers': buyer_specs}
        res.append(trader_specs)
    return res

def run_market_sim_four(trial_id, no_sessions, t, n, supply_range, demand_range, start_time, end_time):

    res = []

    if t == [25, 25, 25, 25]:
        mean_shvr_t = []
        mean_GVWY_t = []
        mean_zic_t = []
        mean_zip_t = []
        seller_spec = [('SHVR', 5), ('GVWY', 5), ('ZIC', 5), ('ZIP', 5)]
        buyer_spec = [('SHVR', 5), ('GVWY', 5), ('ZIC', 5), ('ZIP', 5)]
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
            mean_shvr_t.append(mean_shvr)
            mean_GVWY_t.append(mean_GVWY)
            mean_zic_t.append(mean_zic)
            mean_zip_t.append(mean_zip)
        res.append([mean_shvr_t, mean_GVWY_t, mean_zic_t, mean_zip_t])
    else:
        trader_specs = four_trader_specs(t, n)

        for i, ts in enumerate(trader_specs):
            mean_shvr_t = []
            mean_GVWY_t = []
            mean_zic_t = []
            mean_zip_t = []
            trial_id = trial_id + str(i) + "_shift"
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
                market_session(trial_id, start_time, end_time, ts, order_sched, dump_flags, verbose)
                _, df_profit = df_four(path)
                mean_shvr, mean_GVWY, mean_zic, mean_zip = collect_mean_4(df_profit)
                mean_shvr_t.append(mean_shvr)
                mean_GVWY_t.append(mean_GVWY)
                mean_zic_t.append(mean_zic)
                mean_zip_t.append(mean_zip)
                res.append([mean_shvr_t, mean_GVWY_t, mean_zic_t, mean_zip_t])

            
    return res