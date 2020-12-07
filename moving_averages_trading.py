"""
See Appendix F for resources used
Named this as originally was going to try implement a full algorithmic
trading strategy but would have required to build a backtesting platform.
Instead moved it to Technical Analysis Indicators.
Three Moving Averages, RSI index
@author: William Dagnall
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('fivethirtyeight')



#Plots exponential moving averages of stocks
def buy_sell(ticker):
    data = pd.read_csv(f'stock_dfs/{ticker}.csv')
    data = data.set_index(pd.DatetimeIndex(data['Date'].values))
    short_exponential_moving_averages = data.Close.ewm(span=10, adjust = False).mean()
    print(data)
    #calculate medium exponential moving average
    middle_exponential_moving_averages = data.Close.ewm(span=50, adjust = False).mean()
    
    #Long slow moving averages
    long_moving_exponential_averages = data.Close.ewm(span=200, adjust = False).mean()
    data['Short'] = short_exponential_moving_averages
    data['Middle'] = middle_exponential_moving_averages
    data['Long'] = long_moving_exponential_averages
    
    buy = []
    sell = []
    
    flag_long = False
    flag_short = False
    #Sorting the data to see where is best to buy and sell
    for i in range(0 , len(data)):
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
           buy.append(data['Adj Close'][i])
           sell.append(np.nan)
           flag_short = True
           #Used as signals to see where the EMA's move across the Adj Close Price Line
        elif flag_short == True and data['Short'][i] > data['Middle'][i]:
             sell.append(data['Adj Close'][i])
             buy.append(np.nan)
             flag_short = False
             
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
           buy.append(data['Adj Close'][i])
           sell.append(np.nan)
           flag_long = True
           
        elif flag_long == True and data['Short'][i] < data['Middle'][i]:
            sell.append(data['Adj Close'][i])
            buy.append(np.nan)
            flag_long = False
        else:
            buy.append(np.nan)
            sell.append(np.nan)
             

    data['Buy'] = buy
    data['Sell'] = sell

    
    plt.figure(figsize=(13, 5))
    plt.title(f'{ticker} Adj Close Values', fontsize=16)
    plt.plot(data['Adj Close'], label="Adj Close Price",color = 'red', alpha=0.35)
    
    plt.scatter(data.index, data['Buy'], color = 'green', marker='^', alpha = 0.75)
    plt.scatter(data.index, data['Sell'], color = 'blue', marker = 'v', alpha = 0.75)
    
    plt.plot(short_exponential_moving_averages, label= "Short EMA", color = 'yellow', alpha=0.35)
    plt.plot(middle_exponential_moving_averages, label = "Middle EMA",color = 'green', alpha=0.35)
    plt.plot(long_moving_exponential_averages, label = "Long EMA", color = 'blue', alpha=0.35)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Adj Close Price", fontsize=12)
    plt.legend()
    plt.show()


#buy_sell('NFLX')



def relative_strength_index(ticker):
    
    data = pd.read_csv(f'stock_dfs/{ticker}.csv')
    data = data.set_index(pd.DatetimeIndex(data['Date'].values))

    
        #Use relative strength index RSI, tells whether stock is overbought or not
    change_in_price = data['Adj Close'].diff(1)
    change_in_price = change_in_price.dropna()

    change_upwards = change_in_price.copy()
    change_downwards = change_in_price.copy()
    
    #Positive Values
    change_upwards [change_upwards < 0]= 0
    #Negative Values
    change_downwards [change_downwards >0]= 0
    
    #Timeframe
    days = 14
    #Average gain and loos
    avg_gain = change_upwards.rolling(window=days).mean()
    avg_loss = abs(change_downwards.rolling(window=days).mean())
    
    #RSI
    relative_strength = avg_gain / avg_loss
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))
    

    
    #Plot RSI and various weightings
    plt.figure(figsize=(13, 5))
    plt.title(f'{ticker} Relative Strength Index', fontsize=16)
    plt.plot(relative_strength_index, label = 'Relative Strength Index')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("RSI", fontsize=12)

    plt.axhline(10, linestyle='--', alpha = 0.35, color = 'orange')
    plt.axhline(20, linestyle='--', alpha = 0.35, color = 'red')
    plt.axhline(30, linestyle='--', alpha = 0.35, color = 'green')
    plt.axhline(70, linestyle='--', alpha = 0.35, color = 'green')
    plt.axhline(80, linestyle='--', alpha = 0.35, color = 'red')
    plt.axhline(90, linestyle='--', alpha = 0.35, color = 'orange')
    plt.legend()
    plt.show()
    
    #RSI over 70 indicates security is overbought, under 30 undervalued
    new_dataframe = pd.DataFrame()
    new_dataframe['Adj Close'] = data['Adj Close']
    new_dataframe['RSI'] = relative_strength_index

    
#relative_strength_index('AAPL')

