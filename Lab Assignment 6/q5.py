
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR 
import numpy as np



def AR_to_rmse_mape(p , train ):
    window = p
    model = AR(train, lags=window) # creating model 
    model_fit = model.fit()    # fitting the AR model onto the training dataset.
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]   
    history = [history[i] for i in range(len(history))]
    predictions = []
    for t in range(121):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = yhat
    	predictions.append(yhat) 
    	history.append(obs) 
   
        
    return predictions


# reading the csv file
series= pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')

X = series.values
train = X[:]

third_wave = AR_to_rmse_mape(77, train)

plt.plot(third_wave)
plt.xlabel('Number of days')
plt.ylabel('Number of cases ')
plt.show()

overall_data = []

for i in train:
    overall_data.append(i[0])

for i in third_wave:
    overall_data.append(i[0])
    
    
plt.plot(overall_data)
plt.xlabel('Number of days')
plt.ylabel('Number of cases')
plt.show()



