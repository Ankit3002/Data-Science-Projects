# B20181  , Ankit Pal Singh 
# mobile number -- 9149024234
# Question 4

#importing the modules

import pandas as pd
import math
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR 
import numpy as np



def maker(p , df):
    if p == 1:
        original = df.drop(0, axis = 0)
        lag_p = df.shift(1)
        lag_p = lag_p.drop(0 , axis = 0)
    
    else:
        values = [i for i in range(p)]
        original = df.drop(values , axis = 0)
        lag_p = df.shift(p)
        lag_p = lag_p.drop(values , axis = 0)
    
    corr = original['new_cases'].corr(lag_p['new_cases'])
    T = len(df)  
    
    return corr , T




# reading the csv file
covid = pd.read_csv('daily_covid_cases.csv')

covid = covid[:397]



list = []

for i in range(1,100):
    a , b = maker(i , covid)
    right = 2/(b**0.5)
    
    if(a > right):
        list.append(i)
    

heuristic_value = max(list)
    
print(f'The heuristic value for the optimal number of lags is : {heuristic_value}')

#reading the csv file
series= pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)

train_data, test_data = X[:len(X)-tst_sz], X[len(X)-tst_sz:]




# rmse Calculate root mean squared error
def rmse(actual, predicted):
	sum_error = 0.0
	denom = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
		denom += actual[i]
	mean_error =math.sqrt( sum_error / float(len(actual)))
	denominator = denom / float(len(actual))
	
	return (mean_error/ denominator)*100 
	



# mean_absolute_percentage_error calculate mape      
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 


def AR_to_rmse_mape(p , train , test):
    window = p
    model = AR(train, lags=window) # creating model 
    model_fit = model.fit()    # fitting the AR model onto the training dataset.
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]   
    history = [history[i] for i in range(len(history))]
    predictions = []
    for t in range(len(test)):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = test[t]
    	predictions.append(yhat) 
    	history.append(obs) 
   
        
    return rmse(test,predictions) , mean_absolute_percentage_error(test,predictions)



RMSE , MAPE = AR_to_rmse_mape(heuristic_value, train_data, test_data)


print(f'The RMSE for heuristic value is {RMSE[0]} and the MAPE is {MAPE}.')








