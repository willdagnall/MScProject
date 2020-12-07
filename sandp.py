"""
See Appendix F in code, heavily references pythonprogramming.net
and automate the boring stuff by Al Sweigart, used because it was commonly 
implemented and is very accurate at gaining the stock data into CSV files of the S&P500.
@author: William Dagnall
"""


import matplotlib.pyplot as plt
from matplotlib import style
import bs4 as bs
import datetime as dt
import os
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import pickle
import requests
import yfinance as yf


yf.pdr_override

style.use('ggplot')


def save_sp500_tickers():
    response = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
        
    return tickers



def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
            
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2017, 6, 8)
    end = dt.datetime.now()
    
    
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


#save_sp500_tickers()
#get_data_from_yahoo()

def data_compilation():
    with open ("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    
    main_df = pd.DataFrame()
    
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker.replace('.', '-')))
        df.set_index('Date', inplace=True)
    
        df.rename(columns = {'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)
    
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
    
        if count % 10 == 0:
            print(count)
    
    head = main_df.head()
    print(head)
    main_df.to_csv('sp500_joined_closes.csv')
        
#data_compilation()

