"""
Used in finding pearson coefficient and displaying a Seaborn representation.
See Appendix F for resources used
@author: William Dagnall
"""
import numpy as np
import pandas as pd
 #used to grab the stock prices, with yahoo
import pandas_datareader as web
from datetime import datetime
import matplotlib.pyplot as plt
import multi_asset_stocks
import seaborn as sns

def multi_stock_correlation():
    assets1 = multi_asset_stocks.assets
    print(assets1)
    #Get the stock start date/portfolio
    stockStartDate = '2020-07-15'
    
    #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    
    
    df = pd.DataFrame()
#pull price using iex for each symbol in list defined above
    for stock in assets1:
       df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']


    df.head()
    print(df.head())
    
    corr_df = df.corr(method='pearson')
    corr_df.head().reset_index()
    print("Correlation Coefficients: ")
    print(corr_df)
    
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)]
    sns.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0, mask = mask, linewidths=2.5)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()

#multi_stock_correlation()
    
def multi_stock_correlation1():
    assets1 = multi_asset_stocks.assets1
    print(assets1)
    #Get the stock start date/portfolio
    stockStartDate = '2020-07-15'
    
    #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    
    
    df = pd.DataFrame()
#pull price using iex for each symbol in list defined above
    for stock in assets1:
       df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']


    df.head()
    print(df.head())
        #Finds correlation
    corr_df = df.corr(method='pearson')
    corr_df.head().reset_index()
    
    print(corr_df)
    #Seaborn heatman
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)]
    sns.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0, mask = mask, linewidths=2.5)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    
