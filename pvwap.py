from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tinydb import TinyDB, Query
from datetime import datetime
from datetime import time
from retrying import retry
import matplotlib as plt
import pandas_ta as ta
from time import sleep
from tqdm import tqdm
import datetime as dt
import pandas as pd
import numpy as np
import traceback
import schedule
import requests
import decimal
import logging
import hmac
import time
import keys
import sys
import ftx

# gets theme settings
import plotly.io as plt_io

# creates a custom_dark theme from the plotly_dark template
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

# sets the paper_bgcolor and the plot_bgcolor to a new one
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'

# changes gridline colors if you are modifying background
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

# sets the template to our custom_dark template
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

pairs = ['ETH-PERP', 'BTC-PERP', 'DOT-PERP', 'MKR-PERP', 'MATIC-PERP', 'SOL-PERP']

YOUR_API_KEY = doc.key
YOUR_API_SECRET = doc.skey
API_BASE = 'https://ftx.com/api/'

client = ftx.FtxClient(api_key=YOUR_API_KEY, api_secret=YOUR_API_SECRET)
loline = '____________________________________________________________________'


def Trendicators(pair): # 1hr candles
    # global lrsi, avg_rsi, resolution, price, lvol

    today = dt.datetime.now().date()

    yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
    dbyes1 = np.datetime64('today', 'D') - np.timedelta64(2, 'D')
    dbyes2 = np.datetime64('today', 'D') - np.timedelta64(3, 'D')
    dbyes3 = np.datetime64('today', 'D') - np.timedelta64(4, 'D')
    dbyes4 = np.datetime64('today', 'D') - np.timedelta64(5, 'D')
    dbyes5 = np.datetime64('today', 'D') - np.timedelta64(6, 'D')

    # start = dt.datetime.timestamp(dbyes5.astype(datetime))
    # end = dt.datetime.timestamp(today.astype(datetime))

    end = np.datetime64('now', 's')
    start = end - np.timedelta64(604800, 's')

    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    start = (start - unix_epoch) / one_second
    end = (end - unix_epoch) / one_second

    #start = dt.datetime.fromtimestamp(start)
    #end = dt.datetime.fromtimestamp(end)

    # candle_no = 336 # 14 days back # FTX default is ~1500 values
    resolution = 3600 # 1hr candles
    trend_count = 0
    altc = pair[:-5]
    timenow = dt.datetime.now().time()
    print(f'\n__Start__________Gathering the Trend for {pair}______________{timenow}\n')

    data = client.get_historical_data(pair, resolution, 1580, start, end) # start_time: float ,end_time: float
    df = pd.DataFrame(data=data)

    df['time'] = pd.to_datetime(df['time'] * 1000000, infer_datetime_format=True)
    df.drop('startTime', axis=1, inplace=True)

    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3.0

    df['RSI'] = ta.rsi(close=pd.Series(df['close']), length=7)

    wp = df['hlc3'] * df['volume']
    vwap  = wp.cumsum()
    volume = df['volume'].astype(float)
    vwap /= volume.cumsum()

    df['VWAP'] = vwap

    adxlen = 7
    ADXdf = ta.adx(high=pd.Series(df['high']), low=pd.Series(df['low']), close=pd.Series(df['close']), length=adxlen)
    df['ADX'] = ADXdf[f'ADX_{adxlen}']
    s = df['ADX']
    df['ADX_q'] = s.rolling(7).quantile(.8, interpolation='midpoint')

    return df

def Indicators(pair, df):

    end = np.datetime64('now', 's')
    start = end - np.timedelta64(604800, 's')

    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    start = (start - unix_epoch) / one_second
    end = (end - unix_epoch) / one_second

    altc = pair[:-5]
    resolution = 300 #5 mins candles
    # resolution = 900 #15 mins candles
    data = client.get_historical_data(pair, resolution, 1580, start, end) # start_time: float ,end_time: float
    # Pagination allows you to specify the time range of data to be returned,
    # which also enables you to retrieve more results than are returned by default:
    # 80 more 5min candles
    df = pd.DataFrame(data=data)

    df['time'] = pd.to_datetime(df['time'] * 1000000, infer_datetime_format=True)
    df.drop('startTime', axis=1, inplace=True)
    dates = (df["time"]).dt.date
    df['dates'] = dates

    # df['RSI'] = ta.rsi(close=pd.Series(df['close']), length=17)
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3.0

    volume = df['volume'].astype(float)
    
    today = dt.datetime.now().date()

    yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
    dbyes1 = np.datetime64('today', 'D') - np.timedelta64(2, 'D')
    dbyes2 = np.datetime64('today', 'D') - np.timedelta64(3, 'D')
    dbyes3 = np.datetime64('today', 'D') - np.timedelta64(4, 'D')
    dbyes4 = np.datetime64('today', 'D') - np.timedelta64(5, 'D')
    dbyes5 = np.datetime64('today', 'D') - np.timedelta64(6, 'D')

    df_today = df.loc[df['dates'] == today]
    df_yest = df.loc[df['dates'] == yesterday]
    df_dbyes1 = df.loc[df['dates'] == dbyes1]
    df_dbyes2 = df.loc[df['dates'] == dbyes2]
    df_dbyes3 = df.loc[df['dates'] == dbyes3]
    df_dbyes4 = df.loc[df['dates'] == dbyes4]
    df_dbyes5 = df.loc[df['dates'] < dbyes5]

    # FTX gives about 5 days back in the get_historical_data: for 5min candles
    ref_dates = [df_dbyes5, df_dbyes4, df_dbyes3, df_dbyes2, df_dbyes1, df_yest, df_today]

    
    df['VWAP'] = np.nan
    df['LowrVwap'] = np.nan
    df['UpprVwap'] = np.nan
    
    # Per day: smaller dataframes to pivot VWAP & rolling weighted stdv:
    data_day_list = []
    for i, ref in enumerate(ref_dates):
        wp =ref['hlc3'] * ref['volume']
        volume = ref['volume'].astype(float)
        vwap  = wp.cumsum()
        vwap /= volume.cumsum()
        ref['VWAP'] = vwap

        Sn = ref['volume'] * (ref['hlc3'] - ref['VWAP'].shift(1)) * (ref['hlc3'] - ref['VWAP'])
        Sn = Sn.cumsum()
        dif = abs(ref['hlc3']-ref['VWAP'])/len(ref['hlc3'])
        p_dev = np.sqrt(dif)
        dev = np.sqrt(Sn/(ref['volume'].values).cumsum())

        ref['LowrVwap'] = ref['VWAP'] + ((dev + p_dev*0.16)*-2)
        ref['LowrVwap'].fillna(ref['VWAP'], inplace=True)
        ref['UpprVwap'] = ref['VWAP'] + ((dev + p_dev*0.16)*2)
        ref['UpprVwap'].fillna(ref['VWAP'], inplace=True)

        stdev = ta.stdev(close= pd.Series(df['close']), lenght=int(len(ref)))
        pprofit = (p_dev*2+ref['hlc3']+stdev*2)/ref['hlc3'] # potential movement of the price
        ref['pot_prof'] = pprofit # Used 2 trade away from accumulation

        data_day_list.append(ref) # adding dfs day by day to the list

    df = pd.concat(data_day_list) # concatenates each day with ordered indices

    # Incremental weighted standard deviation (rolling)
    # http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf (part 5)
    # x[i] = hlc3[i], w[i] = volume[i], u[i] - v[i]

    # once is done with each days pivoted rolling values
    df.drop('dates', axis=1, inplace=True) # Removes dates col from Mdf

    return df

def dfMerge(df, dfm5):
    dfm5.index = dfm5['time']
    df.index = df['time']
    merged1 = dfm5.join(df['RSI'], rsuffix='_Added', sort=True)
    merged2 = merged1.join(df['ADX'], rsuffix='_Added', sort=True)
    merged = merged2.join(df['ADX_q'], rsuffix='_Added', sort=True)
    merged.reset_index(inplace = True, drop = True)
    merged['RSI'].fillna(inplace=True, method='ffill')
    merged['ADX'].fillna(inplace=True, method='ffill')
    merged['ADX_q'].fillna(inplace=True, method='ffill')

    pd.set_option('display.max_rows', None)
    mer = merged.drop(merged[merged.time.isnull()].index)
    mer.reset_index(drop=True, inplace=True)

    return mer

def Strategy(mdf, pair):

    # Conditions for triggers:
    mdf['Buy'] = np.zeros(len(mdf))
    mdf['Sell'] = np.zeros(len(mdf))
    mdf['Buy'] = np.where((mdf['Buy'].shift(1) == 0) &
        (mdf['RSI'] < 16) &
        (mdf['pot_prof'] > 0.8) &
        (mdf['low'] <= mdf['LowrVwap']*0.9981), 1, 0) # *0.9981 so is < LowrVwap
    mdf['Sell'] = np.where((mdf['Sell'].shift(1) == 0) &
        (mdf['RSI'] > 60) &
        (mdf['pot_prof'] > 0.8) &
        (mdf['ADX'] > 75) &
        (mdf['high'] >= mdf['UpprVwap']*1.0019), 1, 0) # *1.0019 so is > UpprVwap

    mdf['Buy_Price'] = 1.0016*mdf['close'] # accounting for high slippage
    mdf['Sell_Price'] = 0.9984*mdf['close'] # accounting for high slippage

    mdf['position'] = np.zeros(len(mdf.index))
    mdf['buys'] = np.where((mdf['Buy']==1) & (mdf['Buy'].shift(1)==0), 1, 0)
    mdf['sells'] = np.where((mdf['Sell']==1) & (mdf['Sell'].shift(1)==0), -1, 0)

    mdf['position'] = np.where((mdf['buys'] == 1) &
        (mdf['buys'].shift(1) == 0), 1,0) + np.where((mdf['sells'] == -1) &
        (mdf['sells'].shift(1) == 0), -1,0)

    mdf.drop('hlc3', axis=1, inplace=True)
    x = mdf[(mdf.position == 1) | (mdf.position == -1)]

    #pd.set_option('display.max_rows', None)
    #print(mdf)

    buy_signals, sell_signals = [], []

    bcount = 0
    scount = 0
    for i in range(len(mdf)):
        if mdf['position'].iloc[i] == 1:
            bcount += 1
            buy_signals.append([mdf['position'].iloc[i], mdf['close'].iloc[i], mdf['time'].iloc[i], mdf['Buy_Price'].iloc[i]])
        elif mdf['position'].iloc[i] == -1: #and mdf['position'].iloc[i-1] == 1
            scount += 1
            if bcount >= 1:
                sell_signals.append([mdf['position'].iloc[i], mdf['close'].iloc[i], mdf['time'].iloc[i], mdf['Sell_Price'].iloc[i]])


    print(f'there are {(len(buy_signals))} potential buy signals')
    print(f'there are {(len(sell_signals))} potential sell signals')

    profits = 0

    if len(buy_signals) > 0:
        for i in range(len(x)):
            x['repeatb'] = np.where((x['position'] == 1) & (x['position'].shift(1) == 1), 1, 0)
            x['repeats'] = np.where((x['position'] == -1) & (x['position'].shift(1) == -1), 1, 0)
            y= x.drop(x[x['repeatb'] == 1].index)
            y= y.drop(y[y['repeats'] == 1].index)
        if len(sell_signals) > 0 and y['sells'].iloc[-1] == -1:
            profits = ((y.Sell_Price - y.Buy_Price.shift(1))/y.Buy_Price).values
            y.drop('time', axis=1, inplace=True)
            print(y)
            print(f'realized percent: {profits[1:]}')
        else:
            last = mdf.iloc[-1:]
            z = pd.concat([y,last])
            # mdf[(y.position == 1) | (mdf.Sell_Price == mdf['Sell_Price'].iloc[-1])]
            profits = ((z.Sell_Price - z.Buy_Price.shift(1))/z.Buy_Price).values
            z.drop('time', axis=1, inplace=True)
            print(z)
            if len(profits) == 1:
                print(f'unrealized percent: {profits[1:]}')
            elif len(profits) > 1:
                print(f'realized percent: {profits[1:-1]}')
                print(f'unrealized percent: {profits[-1]}')

        sleep(5)

    return mdf, buy_signals, sell_signals

    #
# Main:
for pair in pairs:
    df = Trendicators(pair)
    dfm5 = Indicators(pair, df)
    mdf = dfMerge(df, dfm5)
    mdf1, buy_signals, sell_signals = Strategy(mdf, pair)
    plot_Pvwap(mdf1, buy_signals, sell_signals, pair)
    sleep(1)
