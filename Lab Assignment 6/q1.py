# B20181  , Ankit Pal Singh 
# mobile number -- 9149024234
# Question 1

# importing modules
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

# reading the csv file
covid = pd.read_csv('daily_covid_cases.csv')


#part_a
# creating the line plot 
series = pd.read_csv('daily_covid_cases.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
plt.show()


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
        
    return original, lag_p
        
    

real , lag_1 = maker(1 , covid)

# part_b
# calculating the correlation
correlation = real['new_cases'].corr(lag_1['new_cases'])

print("The correlation when p = 1 is :")
print(correlation)


# part_c


plt.scatter(real['new_cases'] , lag_1['new_cases'])
plt.xlabel('Original time series')
plt.ylabel('Lag 1 time series')

plt.show()


# part_d

corr_list = []
p_list = []
for i in range(1,7):
    real , lag = maker(i ,covid)
    p_list.append(i)
    corr_list.append(real['new_cases'].corr(lag['new_cases']))


plt.plot(p_list , corr_list)
plt.xlabel('P -- lag')
plt.ylabel('Correlation coefficients at P')
plt.show()


# part_e
del covid["Date"]
sm.graphics.tsa.plot_acf(covid)



