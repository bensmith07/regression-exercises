from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def plot_residuals(x, y, y_hat):
    '''
    this function takes in a set of independent variable values, the corresponding set of 
    dependent variable values, and a set of predictions for the dependent variable. it then displays a plot
    of residuals for the given values. 
    '''
    plt.figure(figsize=(12,8))
    plt.scatter(x, y - y_hat)
    plt.axhline(y = 0, ls = ':')
    plt.show()

def regression_errors(y, y_hat):
     
    SSE = ((y - y_hat) ** 2).sum()
    TSS = SSE_baseline = ((y.mean() - y_hat) ** 2).sum()
    ESS = TSS - SSE
    MSE = mean_squared_error(y, y_hat)
    RMSE = sqrt(MSE)
    
    print(f'SSE: {SSE}')
    print(f'ESS: {ESS}')
    print(f'TSS: {TSS}')
    print(f'MSE: {MSE}')
    print(f'RMSE: {RMSE}')
     
        
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    
    SSE_baseline = ((y - y.mean()) ** 2).sum()
    MSE_baseline = SSE_baseline / len(y)
    RMSE_baseline = sqrt(MSE_baseline)
        
    print(f'Baseline SSE: {SSE_baseline}')
    print(f'Baseline MSE: {MSE_baseline}')
    print(f'Baseline RMSE: {RMSE_baseline}')

    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, y_hat):
    
    SSE = ((y - y_hat) ** 2).sum()
    SSE_baseline = ((y - y.mean()) ** 2).sum()

    if SSE < SSE_baseline:
        return True
    else:
        return False