# PivotedWwap
Pivoted Wvap with rolling weigthed std dev bands per utc day
(Incremental weighted standard deviation (rolling) bands)
https://www.tradingview.com/script/dQF211AS-Multi-Timeframe-VWAP/

# FTX Leveraged Trading

## Skip to
* [Setup](#setup)

## General info

If you find this functions useful please donate here: 

BTC 3BjiRfs2e1bevJHnKfMTi52BKvsq7yVU1u

ETH 0xD528aa90320b1B811B6766159ff727CcF4c14dCc

Leverage on Perp-Trading here is made simpler with FTX! =) You can easily open an account: https://ftx.com/profile#a=2738399

Strategy= Define your own. To use at your own risk/peril. Trades with leveraged pairs as they have lower spreads, but the pairs can be changed to any pair you like.

## Functions available: What it does

Logs in with your API keys from another file (doc.py). Consider a .ini for added security.

Keeps exception logs of any errors with the Rest API/functions/program written in a file with traceback.

Gets 1H & 5m historical data from the API and merges it in a dataframe with OCHLV + VWAP, indicators and other values to then analize it.

Identifies the overal bigger picture trend, 1 Hr candls, Daily GMT pivots for Vwap deviations and more.

You can build (and easily write) your own strategy, could be based on Moving averages, MACD, VWAP, combined or etc. 
# Disclaimer: The strategies presented here just run, most likely it is not profitable (contact me: https://www.linkedin.com/in/carlo-fernandez-benedetto/ to write one)

# To be added:

Goes Long or Short depending on the trend and the strategy chosen, and starts looking for similar pairs faster to trade when volatility increases.

SLs and Trailing TPs with order sizes and database.

Revise and correct the positions in case there is any fault or exception in the process.

Plots and graphs a snapshot of the chart when going long or short indicating the Take profit point and Stop loss prices.

Continuosly checks the open long TP or short orders and cancels them if trend is oposite to the position or any other desired price or indicator condition.


## Technologies
Rest API program created with: Numpy/Pandas
* Pandas Technical Analysis lib pd.ta
* python: 3.8
	
## Setup
To run this project, install python and run in a virtual env the needed libraries can be added with pip install:

```
$ pandas
$ numpy
$ ftx
$ plotly
$ pandas-ta

```
