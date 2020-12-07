"""
See Appendix F for resources used
Uses Markowitz Efficient Frontier to try and optimize users portfolio
@author: William Dagnall
"""
#Import python libraries

from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import multi_asset_stocks
import scipy.optimize as optimizer
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


#Get the stock symbols
#FAANG best performing american stocks
assets = multi_asset_stocks.assets

asset_length = len(assets)
# ASSIGN WEIGHTS to stocks. Weights must add up to 1
weights = multi_asset_stocks.weights_for_optimisation

#Get the stock start date/portfolio
stockStartDate = multi_asset_stocks.stock_start_date

#Ending date
stockEndDate = datetime.today().strftime('%Y-%m-%d')

private_portfolio_investment_weights = [0.3, 0.2, 0.2, 0.1, 0.2]


def get_data(assets, weights):
    
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']

    my_stocks = df
    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label = c)
    
    title = "Adj Close Graph"
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adj. Price')
    plt.legend(my_stocks.columns.values, loc= 'upper left')
    plt.show()
    #Show the df

       
    #Show the daily returns
    returns = df.pct_change()
    print(returns)

    date1 = datetime.today()
    date2 = datetime(2021, 1, 1)
    delta = date2 - date1
    print("Number of days till end of the year: ", delta.days)
    end_of_year_date = delta.days
    
    #Create and show the annualized covariance matrix
    #Covariance matrix shows directional relationship between two assets

    cov_matrix_annual = returns.cov() * end_of_year_date

    
    #Change Weights in multi_asset_stocks
    weights = weights

    #Expected Weighted return of portfolio assets with random weights
    expected_portfolio_return = np.sum(returns.mean()*weights)*end_of_year_date

    
    
    #Calculate the portfolio variance
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

    
    #Calculate the portfolio volatility aka standard deviation
    port_volatility = np.sqrt(port_variance)

    sharpe_ratio = expected_portfolio_return / port_volatility

    


    #Show expected annual return, volatility (risk of the stock), and variance
    print(weights)
    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_volatility = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(expected_portfolio_return, 2) * 100) + '%'
    
    print('Expected Portfolio return: ', percent_ret)
    print('Annual Portfolio risk: ', percent_volatility)
    print('Annual Portfolio Variance', percent_var)
    print("The Portfolio Sharpe Ratio: ", sharpe_ratio)
    
    
    return df[stock]

    #Portfolio Optmization
    
#Uses Markowitz Portfolio Theory to Make a scatter Graph
def port_sim(assets, iterations):
    date1 = datetime.today()
    date2 = datetime(2021, 1, 1)
    delta = date2 - date1
    end_of_year_date = delta.days

    #Collection of data and working out how many days are left in the year
    end_of_year_date = delta.days
    stockStartDate = multi_asset_stocks.stock_start_date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    asset_length = len(assets)
    df = pd.DataFrame()
   
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
    print(df)
    returns = df.pct_change()
    
    expected_portfolio_returns = []
    port_volatility = []


    ret_arr = np.zeros(iterations)
    vol_arr = np.zeros(iterations)
    sharpe_arr = np.zeros(iterations)
    #Random Weighting Generator
    
    for i in range(iterations):
        weights = np.random.dirichlet(np.ones(asset_length), size=1)
        weights = weights[0]
        ret_arr[i] = np.sum(returns.mean()*weights)*end_of_year_date
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * end_of_year_date, weights)))
        expected_portfolio_returns.append(np.sum(returns.mean()*weights)*end_of_year_date)
        port_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * end_of_year_date, weights))))
        
        sharpe_arr[i] = ret_arr[i]/vol_arr[i]
       

    expected_portfolio_returns = np.array(expected_portfolio_returns)
    port_volatility = np.array(port_volatility)
    
    #Finds highest returns and highest volatilitlity
    max_sharpe_returns = ret_arr[sharpe_arr.argmax()]
    max_sharpe_volatility = vol_arr[sharpe_arr.argmax()]
    #Divides the two highest to find the best sharpe ratio
    
    print("Max Sharpe Ratio in array: ", ( max_sharpe_returns/ max_sharpe_volatility))
        #Displays Monte Carlo Portfolio Simulation
    plt.figure(figsize=(18,10))
    plt.scatter(port_volatility, expected_portfolio_returns, c = (expected_portfolio_returns / port_volatility), marker = 'o')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.title("Markowitz Monte Carlo Simulation")
    plt.colorbar(label = 'Sharpe Ratio')
    plt.scatter(max_sharpe_volatility, max_sharpe_returns, c ='red', s=50)
    plt.show()
    
    

    print(weights)


def get_return_volume_sharpe_ratio(weights):
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
    my_stocks = df
    date1 = datetime.today()
    date2 = datetime(2021, 1, 1)
    delta = date2 - date1
    end_of_year_date = delta.days
    weights = np.array(weights)
    print(weights)
    
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * end_of_year_date
    expected_portfolio_return = np.sum(returns.mean()*weights)*end_of_year_date
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
    sharpe_ratio = expected_portfolio_return / port_volatility
    print(sharpe_ratio)
        #Returns a subset of all the data
    return {'return': expected_portfolio_return, 'volatility': port_volatility, 'sharpe ratio': sharpe_ratio}

def minimize_volatility(weights):
   return -get_return_volume_sharpe_ratio(weights)['return']
    #Checks sharpe ratio is negative and makes it positive
def negative_sharpe_ratio(weights):
    return -get_return_volume_sharpe_ratio(weights)['sharpe ratio'] * -1
    #Checks sum of portfolio is = to 1
def check_sum(weights):
    #return 0 if sum of the weights is 1
   return np.sum(weights) -1
    #Aims to reduce volatility

 #Function based off a towards datascience guide
 #This is due to there method working better than the code
 #I had originally made for this part of the project. 
 # See appendix F in report for link.
 
def optimization_function(assets, weights):
    #Uses constraints to set that portfolio equals to 1
    constraints = ({'type' : 'eq', 'fun': check_sum})
    #Refers to the max that one asset can dominate the portfolio
    bounds = tuple((0,0.3) for x in range(asset_length))
    #Initial Values
    initializer = asset_length * [1./asset_length]
    print (initializer)
    print (bounds)
    #Minimize Function
    optimal_sharpe= optimizer.minimize(negative_sharpe_ratio,
                                 initializer,
                                 method = 'SLSQP',
                                 bounds = bounds,
                                 constraints = constraints)
    print(optimal_sharpe)
    get_return_volume_sharpe_ratio(optimal_sharpe.x)

    optimal_sharpe_weights=optimal_sharpe['x'].round(4)
    #Prints Optimal Weights
    print(list(zip(assets,list(optimal_sharpe_weights))))
    optimal_statistics = get_return_volume_sharpe_ratio(optimal_sharpe_weights)
    print(optimal_statistics)
    #Prints SciPy Optimal results Return, Volatility and Sharpe Ratio
    print('Optimal Portfolio Return: ', round(optimal_statistics['return']*100,4))
    print('Optimal Portfolio Volatility: ', round(optimal_statistics['volatility']*100,4))
    print('Optimal Portfolio Sharpe Ratio: ', round(optimal_statistics['sharpe ratio'],4))
    

def check_optimization(assets, weights):
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
    #Calculates expected  returns
    mu = expected_returns.mean_historical_return(df)
    #covariance matrix of asset returns
    S = risk_models.sample_cov(df)
    
    #Optimize for Sharpe Ratio
    efficient_frontier = EfficientFrontier(mu, S)
    weights = efficient_frontier.max_sharpe()
    
    #Helper function to clean weights to remove stocks not needed.
    clean_weights = efficient_frontier.clean_weights()
    print(clean_weights)
    
    efficient_frontier.portfolio_performance(verbose = True)
      
    #Get allocation of each share per stock
    
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    
    latest_prices = get_latest_prices(df)
    weights = clean_weights
    discrete_all = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)
    

    
    allocation, leftover = discrete_all.lp_portfolio()
    print('Discrete Allocation: ', allocation)
    print('Money Left: ${:.2f}'.format(leftover))
    

#negative_sharpe_ratio(weights)
#check_sum(weights)
#check_optimization(assets, weights)
#optimization_function(assets, weights)  
#optimization_function()
#get_data(assets, weights)
#port_sim(assets, 20000)
#get_return_volume_sharpe_ratio(weights)
#minimize_sharpe_ratio()
#port_stats_for_optimization(weights)
