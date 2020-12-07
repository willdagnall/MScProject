"""
Machine Learning implementations.
See Appendix F for resources used
@author: William Dagnall
"""
import numpy as np

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
plt.style.use('bmh')

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential
import math
import pandas_datareader as web
from sklearn.metrics import r2_score

def Decision_Tree_Regression(ticker):
    df = pd.read_csv(f'stock_dfs/{ticker}.csv', index_col = 'Date')
    df.index = pd.to_datetime(df.index)

    #Get number of trading days
    print(df.shape)    
    #Visualise close price data
    
    df = df[['Adj Close']]

    #Create variable to predict x days out into the future
    
    future_prediction_days = 10
    #Create the column to try and predict days
    df['Prediction'] = df[['Adj Close']].shift(-future_prediction_days)


    #Create feature data set and make it numpy array, remove future prediction days
    X = np.array(df.drop(['Prediction'],1))[:-future_prediction_days]
    
    #Create target data set and get all target values
    y = np.array(df['Prediction'])[:-future_prediction_days]

    #Split Data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    #Create Basic Models Decision tree regressor
    tree = DecisionTreeRegressor(max_depth = 10)
    tree.fit(x_train,y_train)
    
    
    #Get last X rows of feature data set
    x_future = df.drop(['Prediction'],1)[:-future_prediction_days]
    x_future = x_future.tail(future_prediction_days)
    x_future = np.array(x_future)

    
    #Show model tree prediction
    
    tree_prediction = tree.predict(x_future)

    print(tree_prediction)
    #Visualise data
    predictions = tree_prediction
    
    valid = df[X.shape[0]:]

    valid['Predictions'] = predictions
    
    
    plt.figure(figsize=(16,10))
    plt.title('Tree Regression Model')
    plt.xlabel('Days')
    plt.ylabel('Adj Close Price')
    plt.plot(df['Adj Close'])
    plt.plot(valid[['Adj Close', 'Predictions']])
    plt.legend(['Original', 'Valid', 'Predictions' ])

    plt.show()
    print("Actual Data", df['Adj Close'].tail(future_prediction_days))
    print("Predictions Data", valid['Predictions'])
    print('Mean Squared Error:', metrics.mean_squared_error(df['Adj Close'].tail(future_prediction_days), predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Adj Close'].tail(future_prediction_days), predictions)))
    print("r2 score", r2_score(df['Adj Close'].tail(future_prediction_days), predictions))
    
#Decision_Tree_Regression('NFLX')

def Random_Forest_Regression(ticker):
    df = pd.read_csv(f'stock_dfs/{ticker}.csv', index_col = 'Date')
    df.index = pd.to_datetime(df.index)


    #Get number of trading days
    print(df.shape)    
    #Visualise close price data
    
    df = df[['Adj Close']]
    
    #Create variable to predict x days out into the future
    
    future_prediction_days = 10
    
    #Create the column to try and predict days
    df['Prediction'] = df[['Adj Close']].shift(-future_prediction_days)
    print(df.tail(4))

    #Create feature data set and make it numpy array, remove future prediction days
    X = np.array(df.drop(['Prediction'],1))[:-future_prediction_days]
    
    #Create target data set and get all target values
    y = np.array(df['Prediction'])[:-future_prediction_days]
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    #N estimators is how many decision trees are used
    #Bootstrap = technique to pick random samples
    #Min_samples_leaf is number of samples till the leaf stops splitting
    random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=7,random_state=42).fit(x_train, y_train)
    
    x_future = df.drop(['Prediction'],1)[:-future_prediction_days]
    x_future = x_future.tail(future_prediction_days)
    x_future = np.array(x_future)    
    predictions = random_forest_regressor.predict(x_future)
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions


    
    
    #Plot Random Forest prediction set
    plt.figure(figsize=(16,10))
    plt.title('Random Forest Regression Model')
    plt.xlabel('Days')
    plt.ylabel('Adj Close Price')
    plt.plot(df['Adj Close'])
    plt.plot(valid[['Adj Close', 'Predictions']])
    plt.legend(['Original', 'Valid', 'Predictions' ])
    plt.show()

    
    print("Actual Data", df['Adj Close'].tail(future_prediction_days))
    print("Predictions Data", valid['Predictions'])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Adj Close'].tail(future_prediction_days), predictions)))
    print("r2 score", r2_score(df['Adj Close'].tail(future_prediction_days), predictions))
#Random_Forest_Regression('NFLX')


def long_short_term_memory(ticker):
    #Call data from CSV
    df = pd.read_csv(f'stock_dfs/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
     

    #Dataframe with just close column
    
    dataframe = df.filter(['Adj Close'])
    dataframe_set = dataframe.values
    #training set size
    training_data = math.ceil(len(dataframe_set) * .75)

    
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scale = scaler.fit_transform(dataframe)

    
    #Training Data Set
    data_train = data_scale[0: training_data, :]
    
    #X and Y data sets
    X_train = []
    y_train = []
    
    #Number of Days used for training the data
    for i in range(80, len(data_train)):
        #Contains past 60 day values
        X_train.append(data_train[i-80:i,0])
        y_train.append(data_train[i, 0])
        
        if i <= 80:
            print(X_train)
            print(y_train)
    
    #Convert datasets
    X_train  = np.array(X_train)
    y_train = np.array(y_train)
    #Reshape the data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_train.shape
    
    #Build LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    lstm_model.add(Dense(25))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(1))
    
    
    #Model compilation optimizer is used to improve loss function, loss function
    #is how well model training performs
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    #Epochs is number of iterations through the data
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    #Testing data with scaled values
    data_test = data_scale[training_data - 80: , :]
    
    X_test = []
    #All test values we want to predict
    y_test = dataframe_set[training_data: , :]
    
    for i in range(80, len(data_test)):
        X_test.append(data_test[i-80:i , 0])
        
        
    X_test = np.array(X_test)
    #End 1 is number of features
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    
    #Get price predictions for LSTM model, want it to contain same values as y test data
    predict = lstm_model.predict(X_test)
    predict = scaler.inverse_transform(predict)
    
    #Root mean squared error to see how accurately model predicts. Lower the better
    rmse = np.sqrt(np.mean(((predict - y_test)**2)))
    print("RMSE = ",rmse)
    
    
    trained_data = dataframe[:training_data]
    valid_data = dataframe[training_data:]
    valid_data['Predict'] = predict
    
    #Plot Data
    plt.figure(figsize=(16,8))
    plt.title('LSTM model')

    plt.plot(trained_data['Adj Close'])
    plt.plot(valid_data[['Adj Close', 'Predict']])
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.legend(['Trained Data', 'Valid', 'Predict'])
    plt.show()
    
    #Valid Data and Predictions Data
    
    print(valid_data)
    
    #Predict one day into the future
    ticker_quote = web.DataReader(f'{ticker}', data_source='yahoo', start = '2020-08-01', end = '2020-08-12')
    new_data_frame = ticker_quote.filter(['Adj Close'])
    last_period_days = new_data_frame[-10:].values
    last_period_days_scaled = scaler.transform(last_period_days)
    
    x_test = []
    #Append period of days
    x_test.append(last_period_days_scaled)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_price = lstm_model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    print("Price Prediction LSTM model: ", predicted_price)
    
    ticker_quote2 = web.DataReader(f'{ticker}', data_source='yahoo', start = '2020-08-12', end = '2020-08-12')
    print(ticker_quote2['Adj Close'])

#long_short_term_memory('AAPL')


def support_vector_rbf_analysis(ticker):
    #Load in data
    df = pd.read_csv(f'stock_dfs/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    

    #Get last row of data
    df = df[['Adj Close']]
      
    
    #Variable that classifies how many days we want to predict
    future_prediction_days = 10
    
    #Create target variable
    df['Predict'] = df[['Adj Close']].shift(-future_prediction_days)

    A = np.array(df.drop(['Predict'],1))
    #Create independent data set
    X = np.array(df.drop(['Predict'],1))

    #Remove future prediction days
    X = X[:-future_prediction_days]
    
    #Dependent data set
    y = np.array(df['Predict'])
    y = y[:-future_prediction_days]

    
    #Split training and test data
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
    

    #SVR using RBF
    rbf_svr = SVR(kernel='rbf', C=10000, gamma = 0.01, epsilon = 2)
    rbf_svr.fit(x_train, y_train)
    #Test: Show coefficient, closer to 1.0 the better
    rbf_coeff = rbf_svr.score(x_test, y_test)
    print("RBF Support Vector Regression coefficient: ", rbf_coeff)
        
        
    #Get last X rows of feature data set
    x_future = df.drop(['Predict'],1)[:-future_prediction_days]
    x_future = x_future.tail(future_prediction_days)
    x_future = np.array(x_future)

   
    #Show RBF SVR
    rbf_svr_predictions = rbf_svr.predict(x_future)
    rbf_predict = rbf_svr_predictions
    rbf_val = df[X.shape[0]:]
    rbf_val['Predict'] = rbf_predict

    
    #Plot RBF Predictions against data
    plt.figure(figsize=(16, 10))
    plt.title('RBF Prediction')
    plt.plot(df['Adj Close'], label= 'Valid Adj Close Data')  
    plt.plot(rbf_val[['Predict']], color = 'orange', label = 'RBF SVR')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    print('RBF SVR predicts: ', rbf_svr_predictions)
    actual_data = df['Adj Close'].tail(future_prediction_days)
    print(actual_data)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Adj Close'].tail(future_prediction_days), rbf_svr_predictions)))
    
    #Print actual Price of stock
#    print(('Actual Adj Close Price: ', comparison_price['Adj Close']))
    
#support_vector_rbf_analysis('NFLX')
    
def polynomial_svr_method(ticker):
    
#Load in data
    df = pd.read_csv(f'stock_dfs/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    

    #Get last row of data
    df = df[['Adj Close']]
      
    
    #Variable that classifies how many days we want to predict
    future_prediction_days = 10
    
    #Create target variable
    df['Predict'] = df[['Adj Close']].shift(-future_prediction_days)
    
    
    #Create independent data set
    X = np.array(df.drop(['Predict'],1))

    #Remove future prediction days
    X = X[:-future_prediction_days]
    
    #Dependent data set
    y = np.array(df['Predict'])
    y = y[:-future_prediction_days]

    
    #Split training and test data
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
    
    #SVR using polynomial
    polynomial_svr = SVR(kernel='poly', C=10000, degree=2.5)
    polynomial_svr.fit(x_train, y_train)
    polynomial_coeff = polynomial_svr.score(x_test, y_test)
    print("Polynomial Support Vector Regression coefficient: ", polynomial_coeff)
    
    #Get last X rows of feature data set
    x_future = df.drop(['Predict'],1)[:-future_prediction_days]
    x_future = x_future.tail(future_prediction_days)
    x_future = np.array(x_future)
    
    
    polynomial_svr_predictions =  polynomial_svr.predict(x_future)
    polynomial_pred = polynomial_svr_predictions
    valid = df[X.shape[0]:]
    valid['Predict'] = polynomial_pred

    #Plot Polynomail Regression
    plt.figure(figsize=(16, 10))
    plt.title('Polynomial Regression')
    plt.plot(df['Adj Close'], color = 'red', label = 'Adj Close Price')
    plt.plot(valid[['Predict']], color = 'green', label = 'Polynomial Reg')
    plt.xlabel('days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    print('Polynomial SVR predicts: ', polynomial_svr_predictions)

    actual_data = df['Adj Close'].tail(future_prediction_days)
    print(actual_data)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Adj Close'].tail(future_prediction_days), polynomial_svr_predictions)))
    
#polynomial_svr_method('AAPL')


def linear_regression_method(ticker):
    
    #Load in data
    df = pd.read_csv(f'stock_dfs/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    

    #Get last row of data
    df = df[['Adj Close']]
      
    
    #Variable that classifies how many days we want to predict
    future_prediction_days = 10
    
    #Create target variable
    df['Predict'] = df[['Adj Close']].shift(-future_prediction_days)

    
    #Create independent data set
    X = np.array(df.drop(['Predict'],1))

    #Remove future prediction days
    X = X[:-future_prediction_days]
    
    #Dependent data set
    y = np.array(df['Predict'])
    y = y[:-future_prediction_days]

    
    #Split training and test data
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
    
    #Support Vector Regression Models
    linear_regression = LinearRegression()
    linear_regression.fit(x_train , y_train)
    linear_reg_coeff = linear_regression.score(x_test, y_test)
    print("Linear Regression: ", linear_reg_coeff)
    
    #Get last X rows of feature data set
    x_future = df.drop(['Predict'],1)[:-future_prediction_days]
    x_future = x_future.tail(future_prediction_days)
    x_future = np.array(x_future)
    
    
    #Show Linear Regression prediction
    linear_regression_prediction = linear_regression.predict(x_future)
    #Visualise data
    predictions = linear_regression_prediction
    valid = df[X.shape[0]:]
    valid['Predict'] = predictions

    #Plot Linear Regression
    plt.figure(figsize=(16, 10))
    plt.title('Linear Regression Predictions')
    plt.plot(df['Adj Close'], color = 'red', label = 'Adj Close Price')
    plt.plot(valid[['Predict']], color = 'green', label = 'Linear Regression')
    plt.xlabel('days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    print('Linear Regression predicts: ', linear_regression_prediction)

    actual_data = df['Adj Close'].tail(future_prediction_days)
    print(actual_data)


#linear_regression_method('NFLX')    


def linear_svr_method(ticker):

    
    #Load in data
    df = pd.read_csv(f'stock_dfs/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    

    #Get last row of data
    df = df[['Adj Close']]
      
    
    #Variable that classifies how many days we want to predict
    future_prediction_days = 10
    
    #Create target variable
    df['Predict'] = df[['Adj Close']].shift(-future_prediction_days)

    
    #Create independent data set
    X = np.array(df.drop(['Predict'],1))

    #Remove future prediction days
    X = X[:-future_prediction_days]
    
    #Dependent data set
    y = np.array(df['Predict'])
    y = y[:-future_prediction_days]

    
    #Split training and test data
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
    
    #SVR using linear
    linear_svr = SVR(kernel='linear', C=100)
    linear_svr.fit(x_train, y_train)
    #Test: Show coefficient, closer to 1.0 the better
    linear_coeff = linear_svr.score(x_test, y_test)
    print("Linear Support Vector Regression coefficient: ", linear_coeff)
    
    #Get last X rows of feature data set
    x_future = df.drop(['Predict'],1)[:-future_prediction_days]
    x_future = x_future.tail(future_prediction_days)
    x_future = np.array(x_future)
    
    
    #Show Linear SVR
    linear_svr_prediction = linear_svr.predict(x_future)
    linear_pred = linear_svr_prediction
    valid = df[X.shape[0]:]
    valid['Predict'] = linear_pred

    #Plots graph
    plt.figure(figsize=(16, 10))
    plt.title('Linear SVR Stock Prediction')
    plt.plot(df['Adj Close'], color = 'red', label = 'Adj Close Price')
    plt.plot(valid[['Predict']], color = 'green', label = 'Linear SVR')
    plt.xlabel('days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    print('Linear SVR predicts: ', linear_svr_prediction)
    
    
    actual_data = df['Adj Close'].tail(future_prediction_days)
    print(actual_data)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Adj Close'].tail(future_prediction_days), linear_svr_prediction)))


#linear_svr_method('NFLX')
