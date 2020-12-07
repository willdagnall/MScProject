"""
See Appendix F in report for resources used
GUI Class
@author: William Dagnall
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.animation as animation
from pandas_datareader import data as web
import pandas as pd
from datetime import datetime
import sandp
import multi_asset_stocks
import Pearson_Coefficient
import portfolio_optimization
import MachineLearning
import moving_averages_trading

LARGE_FONT = ("Verdana", 12)
NORMAL_FONT = ("Verdana", 10)
style.use("ggplot")

f = Figure()   
a = f.add_subplot(111)

j = Figure()
b = j.add_subplot(111)

k = Figure()
d = k.add_subplot(111)

l = Figure()
e = l.add_subplot(111)

m = Figure()
g = m.add_subplot(111)

ticker_for_single_asset = multi_asset_stocks.single_asset_for_stock 

#Calls methods from different classes

def get_portfolio_best_optimisation():
    portfolio_optimization.check_optimization(multi_asset_stocks.assets, multi_asset_stocks.weights_for_optimisation)


def get_polynomial():
    MachineLearning.polynomial_svr_method(multi_asset_stocks.single_asset)
    
def get_linear_reg():
    MachineLearning.linear_regression_method(multi_asset_stocks.single_asset)
    
def get_linear_svr():
    MachineLearning.linear_svr_method(multi_asset_stocks.single_asset)
 

def get_r_s_i():
    moving_averages_trading.relative_strength_index(multi_asset_stocks.single_asset)

def get_lstm_neural_network():
    MachineLearning.long_short_term_memory(multi_asset_stocks.single_asset)

def get_moving_averages():
    moving_averages_trading.buy_sell(multi_asset_stocks.single_asset)

def get_decision_tree_regression():
    MachineLearning.Decision_Tree_Regression(multi_asset_stocks.single_asset)

def get_linear_regression():
    MachineLearning.Random_Forest_Regression(multi_asset_stocks.single_asset)

def get_markowitz_portfolio_graph():
    portfolio_optimization.port_sim(multi_asset_stocks.assets, 2000)

def get_optimization_of_assets():
    portfolio_optimization.optimization_function(multi_asset_stocks.assets,
                                                 multi_asset_stocks.weights_for_optimisation)
    
def get_support_vector_regression():
    MachineLearning.support_vector_rbf_analysis(multi_asset_stocks.single_asset)
    
def web_scrape():
    sandp.get_data_from_yahoo()
    sandp.data_compilation()
    
def get_pearson_coefficient():
    Pearson_Coefficient.multi_stock_correlation()
    
def get_pearson_coefficient1():
    Pearson_Coefficient.multi_stock_correlation1()
    
def animate(i) :
    
    assets =  multi_asset_stocks.assets
    stockStartDate = '2018-07-15'
    
    #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    
    #Create a dataframe to store close price of stocks
    
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
        
        a.clear()
    #Visually show the df
    title = 'Portfolio Adj. Close Price History'
    a.set_title(title)
    a.set_xlabel('Date')
    a.set_ylabel('Adj Close Price')
        #bbox_to_anchor=tuple=(0, 1.02, 1, .102), loc=3, ncol=2, borderaxespad=0)
    
    # Get the stocks
    my_stocks = df
    #Create and plot the graph c = column
    for c in my_stocks.columns.values:
        a.plot(my_stocks[c], label = c)

    a.legend()
    print(df.tail(1))
    
    
def animate2(i):
    assets1 = multi_asset_stocks.assets1
  
    #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    stockStartDate = '2018-07-15'
    #Create a dataframe to store close price of stocks
    
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    
    for stock in assets1:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
        
        b.clear()
    #Visually show the df
    title = 'Portfolio Adj. Close Price History'
    b.set_title(title)
    b.set_xlabel('Date')
    b.set_ylabel('Adj Close Price')
        #bbox_to_anchor=tuple=(0, 1.02, 1, .102), loc=3, ncol=2, borderaxespad=0)
    
    # Get the stocks
    my_stocks = df
    #Create and plot the graph c = column
    for c in my_stocks.columns.values:
        b.plot(my_stocks[c], label = c)

    b.legend()
    print(df.tail(1))

def animate3(i):
    assets2 =['AAPL']
  
    #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    stockStartDate = '2018-07-15'
    #Create a dataframe to store close price of stocks
    
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    
    for stock in assets2:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
        
        d.clear()
    #Visually show the df
    title = 'Portfolio Adj. Close Price History'
    d.set_xlabel('Date')
    d.set_ylabel('Adj Close Price')
    d.set_title(title)
        #bbox_to_anchor=tuple=(0, 1.02, 1, .102), loc=3, ncol=2, borderaxespad=0)
    
    # Get the stocks
    my_stocks = df
    #Create and plot the graph c = column
    for c in my_stocks.columns.values:
        d.plot(my_stocks[c], label = c)

    d.legend()
    print(df.tail(1))
    
def animate4(i):
    assets3 = ticker_for_single_asset
  
        #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    stockStartDate = '2018-07-15'
    #Create a dataframe to store close price of stocks
    
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    
    for stock in assets3:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
        
        e.clear()
    #Visually show the df
    title = 'Portfolio Adj. Close Price History'
    e.set_title(title)
    e.set_xlabel('Date')
    e.set_ylabel('Adj Close Price')
        #bbox_to_anchor=tuple=(0, 1.02, 1, .102), loc=3, ncol=2, borderaxespad=0)
    
    # Get the stocks
    my_stocks = df
    #Create and plot the graph c = column
    for c in my_stocks.columns.values:
        e.plot(my_stocks[c], label = c)

    e.legend()
    print(df.tail(1))
    
def animate5(i):
    assets4 =['GOOG']
  
        #Ending date
    stockEndDate = datetime.today().strftime('%Y-%m-%d')
    stockStartDate = '2018-07-15'
    #Create a dataframe to store close price of stocks
    
    df = pd.DataFrame()
    
    #Store adjusted close price of stock into the df
    
    for stock in assets4:
        df[stock] = web.DataReader(stock, data_source ='yahoo', start = stockStartDate, end = stockEndDate) ['Adj Close']
        
        g.clear()
    #Visually show the df
    title = 'Portfolio Adj. Close Price History'
    g.set_title(title)
    g.set_xlabel('Date')
    g.set_ylabel('Adj Close Price')
        #bbox_to_anchor=tuple=(0, 1.02, 1, .102), loc=3, ncol=2, borderaxespad=0)
    
    # Get the stocks
    my_stocks = df
    #Create and plot the graph c = column
    for c in my_stocks.columns.values:
        g.plot(my_stocks[c], label = c)

    g.legend()
    print(df.tail(1))
    
class Portfolioapp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        
        tk.Tk.wm_title(self, "Portfolio App")
        
        #Window
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        menu = tk.Menu(container)
        filem = tk.Menu(menu, tearoff=0)
        filem.add_command(label="update Settings")

        #, command = lambda: popupmsg("Not supported!"))
        filem.add_separator()
        menu.add_cascade(label="File", menu=filem)
        

    
        
        tk.Tk.config(self, menu= menu)
        #Lists all the pages within the Robo-advisory
        for Fram in (StartPage, Index_Page, Single_Stock_Index, Multi_Stock_Index,
                     FAANG_Stocks, Multi_Stock_View, Web_Scraping_Function, Portfolio_Optimization_Of_Stocks,
                     Algorithmic_Trading, Machine_Learning, Apple_Stock_Show, IBM_Stock_Show, Google_Stock_Show):
            frame = Fram(container, self)
            
            self.frames[Fram] = frame
            #Stickey parameter is north south east west
            frame.grid(row=0, column=0, sticky="nsew")
            
            
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()





        #StartPage inherits from frame
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        #Creates start page
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Robo-advisory trading app", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        
        #Use lambda to correct display and navigation page
        button2 = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        button2.pack()
    


class Index_Page(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Index Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Title_Page_Button = ttk.Button(self, text="Title Page", width=20,
                             command=lambda: controller.show_frame(StartPage))
        
        Title_Page_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        #Creates buttons for the Tkinter page
        Stock_View_Button = ttk.Button(self, text="Stock Viewer",width=20,
                             command=lambda: controller.show_frame(Single_Stock_Index))
        Stock_View_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Multi_Stock_Button = ttk.Button(self, text="Multi-Stock Viewer",width=20,
                             command=lambda: controller.show_frame(Multi_Stock_Index))
        Multi_Stock_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Web_Scraping_Function_Button = ttk.Button(self, text="Web Scrape S&P 500",width=20,
                                           command=lambda: controller.show_frame(Web_Scraping_Function))
        Web_Scraping_Function_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Algorithmic_Trading_Button = ttk.Button(self, text="Stock Analysis",width=20,
                                           command=lambda: controller.show_frame(Algorithmic_Trading))
        Algorithmic_Trading_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Machine_Learning_Button = ttk.Button(self, text="Machine Learning",width=20,
                                             command=lambda: controller.show_frame(Machine_Learning))
        Machine_Learning_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Portfolio_Optimization_Button = ttk.Button(self, text = "Portfolio Optimization",width=20,
                                                   command= lambda: controller.show_frame(Portfolio_Optimization_Of_Stocks))
        Portfolio_Optimization_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        


class Single_Stock_Index(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Single Stock Index", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Index_Button = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        Index_Button.pack()
        
        Apple_Stocks_Button = ttk.Button(self, text="Apple Stock",width=20,
                             command=lambda: controller.show_frame(Apple_Stock_Show))
        Apple_Stocks_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
 
        IBM_Stocks_Button = ttk.Button(self, text=f"{ticker_for_single_asset}",width=20,
                             command=lambda: controller.show_frame(IBM_Stock_Show))
        IBM_Stocks_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Google_Stocks_Button = ttk.Button(self, text="Google Stock",width=20,
                             command=lambda: controller.show_frame(Google_Stock_Show))
        Google_Stocks_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        
        #Plots Single Stocks
class Apple_Stock_Show(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Apple Stock", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        
        Single_Stock_Index_Button = ttk.Button(self, text="Stock Index Page" ,
                             command=lambda: controller.show_frame(Single_Stock_Index))
        Single_Stock_Index_Button.pack()

        canvas = FigureCanvasTkAgg(k, self)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        


class IBM_Stock_Show(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=f"{ticker_for_single_asset}", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        
        Index_Button = ttk.Button(self, text="Stock Index Page",
                             command=lambda: controller.show_frame(Single_Stock_Index))
        Index_Button.pack()

        canvas = FigureCanvasTkAgg(l, self)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand = True)


class Google_Stock_Show(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Google Stock", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        
        Index_Button = ttk.Button(self, text="Stock Index Page",
                             command=lambda: controller.show_frame(Single_Stock_Index))
        Index_Button.pack()

        canvas = FigureCanvasTkAgg(m, self)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand = True)



    #Plots Multi-stock portfolios
class Multi_Stock_Index(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Multi Stock Index", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Index_Button = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        Index_Button.pack()
        
        FAANG_Stocks_Button = ttk.Button(self, text="FAANG Stocks",width=20,
                             command=lambda: controller.show_frame(FAANG_Stocks))
        FAANG_Stocks_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Test_Multi_Stock_Button = ttk.Button(self, text="Multi-Stock",width=20,
                             command=lambda: controller.show_frame(Multi_Stock_View))
        Test_Multi_Stock_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        

class FAANG_Stocks(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="FAANG Stocks", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        
        Multi_Stock_Index_Button = ttk.Button(self, text="Multi Stock Index",
                             command=lambda: controller.show_frame(Multi_Stock_Index))
        Multi_Stock_Index_Button.pack()
        
        
        
        pearson_coefficient_label = tk.Label(self, text="Click button to see the pearson coefficient", font=LARGE_FONT)
        pearson_coefficient_label.pack()
        pearson_coefficient_button = tk.Button(self,text="Pearson Coefficient of Stocks",
                                        command=lambda: get_pearson_coefficient())
        pearson_coefficient_button.pack()
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        
        
        
        
    

class Multi_Stock_View(tk.Frame):
    
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Other Stocks", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Multi_Stock_Index_Button = ttk.Button(self, text="Multi Stock Index",
                             command=lambda: controller.show_frame(Multi_Stock_Index))
        Multi_Stock_Index_Button.pack()
        
        pearson_coefficient_label = tk.Label(self, text="Click button to see the pearson coefficient", font=LARGE_FONT)
        pearson_coefficient_label.pack()
        pearson_coefficient_button = tk.Button(self,text="Pearson Coefficient of Stocks",
                                        command=lambda: get_pearson_coefficient1())
        pearson_coefficient_button.pack()
        canvas = FigureCanvasTkAgg(j, self)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        


class Web_Scraping_Function(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Web Scraping S&P 500", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Index_Button = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        Index_Button.pack()
        

 
        
        web_scrape_label = tk.Label(self, text="Click button to Web Scrape", font=LARGE_FONT)
        web_scrape_label.pack()
        activate_web_scrape = tk.Button(self,text="Commence Web Scrape",
                                        command=lambda: web_scrape())
        
        activate_web_scrape.pack(side="top", fill='both', expand=True, padx=4, pady=4)
   
        

        
        
        
        #Class used to make frame for portfolio optimization functions
class Portfolio_Optimization_Of_Stocks(tk.Frame):
    
     def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Portfolio Optimisation", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Index_Button = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        Index_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show Markowitz portfolio", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        portfolio_scatter_button = tk.Button(self,text="Show Markowitz Portfolio Theory",
                                        command=lambda: get_markowitz_portfolio_graph())
        portfolio_scatter_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show optimization function in terminal", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        optimization_button = tk.Button(self,text="Optimization Function ",
                                        command=lambda: get_optimization_of_assets())
        optimization_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show PyPortfolio optimization in terminal", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        best_optimization_button = tk.Button(self,text="Best Sharpe Ratio PyPortfolio Optimization",
                                        command=lambda: get_portfolio_best_optimisation())
        best_optimization_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        #Class used to make frame for Technical Analysis functions
class Algorithmic_Trading(tk.Frame):
    
    def __init__(self, parent, controller) :
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Moving Averages", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        Index_Button = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        Index_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        moving_averages_button = tk.Button(self,text="Show Moving Averages",
                                        command=lambda: get_moving_averages())
        moving_averages_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)


        rsi_button = tk.Button(self,text="Show Relative Strength Index",
                                        command=lambda: get_r_s_i())
        rsi_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
    #Class used to make frame for machine learning functions
class Machine_Learning(tk.Frame):
    def __init__(self, parent, controller) :
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Machine Learning", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
                  
        Index_Button = ttk.Button(self, text="Index Page",
                             command=lambda: controller.show_frame(Index_Page))
        Index_Button.pack(side="top", fill='both', expand=True, padx=4, pady=4)  
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show Decision Tree Regression", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        decision_tree_button = tk.Button(self,text="Show Decision Tree Regression",
                                        command=lambda: get_decision_tree_regression())
        decision_tree_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show Random Forest Regression", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        random_forest_button = tk.Button(self,text="Show Random Forest Regression",
                                        command=lambda: get_linear_regression())
        random_forest_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)

        Display_portfolio_scatter = tk.Label(self, text="Click below to show RBF Vector", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        support_vector_button = tk.Button(self,text="Show RBF Vector Regression",
                                        command=lambda: get_support_vector_regression())
        support_vector_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show LSTM Neural Network", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        lstm_neural_button = tk.Button(self,text="Show LSTM neural network",
                                        command=lambda: get_lstm_neural_network())
        lstm_neural_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click below to show Polynomial SVR", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        polynomial_button = tk.Button(self,text="Show Polynomial SVR Regression",
                                        command=lambda: get_polynomial())
        polynomial_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click to show Linear Regression", font=LARGE_FONT)
        Display_portfolio_scatter.pack()
        linear_reg_button = tk.Button(self,text="Show Linear Regression",
                                        command=lambda: get_linear_reg())
        linear_reg_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        Display_portfolio_scatter = tk.Label(self, text="Click to show Linear SVR", font=LARGE_FONT)
        Display_portfolio_scatter.pack(side="top", fill='both', expand=True, padx=4, pady=4)
        
        linear_reg__svr_button = tk.Button(self,text="Show Linear SVR Regression",
                                        command=lambda: get_linear_svr())
        linear_reg__svr_button.pack(side="top", fill='both', expand=True, padx=4, pady=4)


#Used to refresh data appearing in graph
app = Portfolioapp()
app.geometry("1200x680")
ani = animation.FuncAnimation(f, animate, interval=100000)
anim = animation.FuncAnimation(j, animate2, interval=500000)
anima = animation.FuncAnimation(k, animate3, interval=500000)
animat = animation.FuncAnimation(l, animate4, interval=500000)
animati = animation.FuncAnimation(m, animate5, interval=500000)
app.mainloop()
