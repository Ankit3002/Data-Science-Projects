# B20181  , Ankit Pal Singh 
# mobile number -- 9149024234
# Question 2


# importing modules
import pandas as pd
import math
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR 
import numpy as np


# part_a

# reading the csv file
series= pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = series[:len(X)-tst_sz], series[len(X)-tst_sz:]

# printing the train sets
train.plot()
plt.show()
# printing the test sets
test.plot()
plt.show()



train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# part_b

# lags = 5 

# window are the values for lags 

window = 5
model = AR(train, lags=window) # creating model 
model_fit = model.fit()    # fitting the AR model onto the training dataset.
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]   
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)

print(" the coefficient are :")	
print(coef)
# part b_(i)

plt.scatter(test,predictions)
plt.title('Scatter plot between actual values and predicted values')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()  

# part b_(ii)

plt.plot(test, predictions)
plt.title('Line plot between actual values and predicted values')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

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


# part b_(iii)

rmse_value = rmse(test, predictions)
mape  = mean_absolute_percentage_error(test,predictions)


print(f'The Rmse vaue in % is {rmse_value}')
     
print(f'The value of MAPE is : {mape}')  
    
    


    