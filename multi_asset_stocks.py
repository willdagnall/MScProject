"""
Variable Class for different functions.
@author: William Dagnall
"""
import numpy as np
from datetime import datetime as dt

assets = ['FB','AMZN','AAPL','TSLA','GOOG']
assets1 = ['TSLA', 'IBM', 'MCD', 'NKE']
single_asset_for_stock = ["AMZN"]
single_asset = 'AAPL'

stock_start_date = '2018-07-15'
stockEndDate = dt.today().strftime('%Y-%m-%d')

#Must add up to 1
weights_for_optimisation = np.array([0.2,0.2,0.2,0.2,0.2])








#List of different SandP 500 stocks below

assetsMMM_ADBE = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE']
assetsAMD_AKAM = ['AMD', 'AAP', 'AES', 'AFL', 'A','APD', 'AKAM']
assetsALK_LNT = ['ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'LNT']
assetsALK_AEE = ['ALL', 'GOOGL', 'GOOG','MO', 'AMZN', 'AMCR', 'AEE']
assetsAAL_AMP = ['AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP']
assetsABC_ANTM = ['ABC', 'AME','AMGN', 'APH', 'ADI', 'ANSS', 'ANTM']
assetsAON_APTV = ['AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV']
assetsADM_ADSK = ['ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK']
assetsADP_BAC = ['ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL','BAC']
assetsBK_BIIB = [ 'BK', 'BAX', 'BDX', 'BRK-B', 'BBY', 'BIO', 'BIIB']
assetsBLK_BMY = ['BLK', 'BA', 'BKNG', 'BWA', 'BXP','BSX', 'BMY']
assetsAVGO_CPB = ['AVGO', 'BR', 'BF-B', 'CHRW', 'COG', 'CDNS', 'CPB']
assetsCOF_CBOE = ['COF', 'CAH', 'KMX', 'CCL','CARR', 'CAT', 'CBOE']
assetsCBRE_CERN = ['CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CTL', 'CERN']
assetsCF_CHD = ['CF', 'SCHW', 'CHTR','CVX', 'CMG', 'CB', 'CHD']
assetsCI_CTXS = ['CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS']
assetsCLX_CMCSA = ['CLX', 'CME', 'CMS','KO', 'CTSH', 'CL', 'CMCSA']
assetsCMA_COO = ['CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO']
assetsCPRT_CSX = ['CPRT', 'GLW', 'CTVA','COST', 'COTY', 'CCI', 'CSX',]
assetsCMI_DE = [ 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE']
assetsDAL_DFS = ['DAL', 'XRAY', 'DVN','DXCM', 'FANG', 'DLR', 'DFS']
assetsDISCA_DPZ = ['DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DPZ']
assetsDOV_DXC = ['DOV', 'DOW', 'DTE','DUK', 'DRE', 'DD', 'DXC']
assetsETFC_EW = ['ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW']
assetsEA_EQR = ['EA', 'EMR', 'ETR', 'EOG','EFX', 'EQIX', 'EQR']
assetsESS_EXPE = ['ESS', 'EL', 'EVRG', 'ES', 'RE', 'EXC', 'EXPE']
assetsEXPD_FRT = ['EXPD', 'EXR', 'XOM', 'FFIV','FB', 'FAST', 'FRT']
assetsFDX_FLT = ['FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT']
assetsFLIR_FBHS = ['FLIR', 'FLS', 'FMC', 'F','FTNT', 'FTV', 'FBHS',]
assetsFOXA_IT = ['FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT']
assetsGD_GL = ['GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL']
assetsGPN_HIG = ['GPN', 'GS', 'GWW', 'HRB', 'HAL', 'HBI', 'HIG']
assetsHAS_HPE = ['HAS', 'HCA', 'PEAK', 'HSIC','HSY', 'HES', 'HPE']
assetsHLT_HST = ['HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL', 'HST']
assetsHWM_IDXX = ['HWM', 'HPQ', 'HUM', 'HBAN', 'HII','IEX', 'IDXX']
assetsINFO_ICE = ['INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE']
assetsIBM_IVZ = ['IBM', 'IP', 'IPG', 'IFF', 'INTU','ISRG', 'IVZ']
assetsIPGP_SJM = ['IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM']
assetsJNJ_KEY = ['JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY']
assetsKEYS_KHC = ['KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC']
assetsKR_LVS = ['KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW','LVS']
assetsLEG_LYV = ['LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV']
assetsLKQ_MRO = ['LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MRO']
assetsMPC_MA = ['MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA']
assetsMKC_MET = ['MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET']
assetsMTD_MHK = ['MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK']
assetsTAP_MSI = ['TAP', 'MDLZ', 'MNST', 'MCO', 'MS', 'MOS', 'MSI']
assetsMSCI_NWL = ['MSCI', 'MYL', 'NDAQ', 'NOV', 'NTAP', 'NFLX', 'NWL']
assetsNEM_NI = ['NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI'] 
assetsNBL_NRG = ['NBL', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG']
assetsNUE_OMC = ['NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC']
assetsOKE_PAYX = ['OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX']
assetsPAYC_PRGO = ['PAYC', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO']
assetsPFE_PPG = ['PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'PPG']
assetsPPL_PEG = [ 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG']
assetsPSA_DGX = ['PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX']
assetsRL_RF = ['RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF']
assetsRSG_ROST = ['RSG','RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', ]
assetsRCL_SEE = ['RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE']
assetsSRE_SNA = ['SRE','NOW', 'SHW', 'SPG', 'SWKS', 'SLG', 'SNA']
assetsSO_SYK = ['SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK']
assetsSIVB_TTWO = ['SIVB','SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO']
assetsTPR_TXN = ['TPR', 'TGT', 'TEL', 'FTI', 'TDY', 'TFX', 'TXN']
assetsTXT_TDG = ['TXT','TMO', 'TIF', 'TJX', 'TSCO', 'TT', 'TDG']
assetsTRV_ULTA = ['TRV', 'TFC', 'TWTR', 'TYL', 'TSN', 'UDR', 'ULTA']
assetsUSB_UPS = ['USB','UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS']
assetsURI_ = ['URI', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR']
assetsVRSN_VNO = ['VRSN','VRSK', 'VZ', 'VRTX', 'VIAC', 'V', 'VNO']
assetsVMC_WM = ['VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS', 'WM' ]
assetsWAT_WU = ['WAT','WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU']
assetsWRK_XEL = ['WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL']
assetsXRX_ZION = ['XRX','XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION']
assetsZTS = ['ZTS']
