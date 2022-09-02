# Ankit Pal Singh , B20181 , 9149024234
#Question 1

# importing modules
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# reading the csv file
csv = pd.read_csv('abalone.csv')


# splitting the data into training set and testing set
train,test = train_test_split(csv,test_size=0.3, random_state=42,shuffle = True)


# converting the dataframes into csv
train.to_csv('abalone-train.csv',index = False)
test.to_csv('abalone-test.csv', index = False)


train_data = pd.read_csv('abalone-train.csv')
test_data = pd.read_csv('abalone-test.csv')

attr = [i for i in train.columns]



def regression(trains,tests):
    # acc have correlation coefficients 
    acc = {}
    for i in attr:
        if (i !='Rings'):
        
            acc.update({i:train_data['Rings'].corr(train_data[i])})

    max_corr_attr = max(zip(acc.values(), acc.keys()))[1]  # max_corr_attr is the attribute which has maximum correlation with Rings attribute
    
    # Reshaping the training data
    X = np.array(train_data[max_corr_attr]).reshape(-1,1)
    Y = np.array(train_data['Rings']).reshape(-1,1)
    
    # Reshaping the testing data
    x = np.array(test_data[max_corr_attr]).reshape(-1,1)
    
    
    reg = LinearRegression().fit(X ,Y) 
    pred = reg.predict(x)
    return pred,max_corr_attr,reg
    


print('question - 1')
print()

predicted , max_attr , reg = regression(train_data, test_data)


plt.scatter(train_data[max_attr],train_data['Rings'])


# part a      

#plotting the best fit line 
h = np.linspace(0,1)
h = np.array(h).reshape(-1,1)
v = reg.predict(h)
plt.scatter(train_data[max_attr], train_data['Rings'])
plt.plot(h, v, color = "green")
plt.xlabel(max_attr)
plt.ylabel('Rings')
plt.title('Q1: best linear fit')
plt.show()







# part b

y_train_pred = reg.predict(np.array(train_data["Shell weight"]).reshape(-1, 1))
rmse_train = (mean_squared_error(train_data['Rings'], y_train_pred)) ** 0.5
print("The rmse for training data is", round(rmse_train, 3))



tot = train_data['Rings'].mean()
acc = round((1-(rmse_train/tot))*100,3)
print('and the  accuracy is ')
print(acc)



# part c

y_test_pred = reg.predict(np.array(test_data["Shell weight"]).reshape(-1, 1))
rmse_test = (mean_squared_error(test_data['Rings'].to_numpy(), y_test_pred)) ** 0.5
print("The rmse for testing data is", round(rmse_test, 3))

tot = test_data['Rings'].mean()
acc = round((1-(rmse_test/tot))*100,3)
print('and the  accuracy is ')
print(acc)

# part d

plt.scatter(test_data['Rings'].to_numpy(), y_test_pred,color = 'r')
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.show()



# question 2
print()
print('question -2')

X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, train.shape[1] - 1].values
X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, test.shape[1] - 1].values

print('a:')
reg_train = LinearRegression().fit(X_train, Y_train)
rmse_train = (mean_squared_error(Y_train, reg_train.predict(X_train))) ** 0.5
print("The rmse for training data is", round(rmse_train, 3))

print('b:')
reg_test = LinearRegression().fit(X_test, Y_test)
rmse_test = (mean_squared_error(Y_test, reg_test.predict(X_test))) ** 0.5
print("The rmse for testing data is", round(rmse_test, 3))

plt.scatter(Y_test, reg_test.predict(X_test),color = 'r')
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Multivariate linear regression model')
plt.show()


# question 3
print()
print('question - 3')

P = [2, 3, 4, 5]

print('a:')
X = np.array(train_data[max_attr]).reshape(-1, 1)
RMSE = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (mean_squared_error(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Univariate non-linear regression model")
plt.show()

print('b:')
RMSE = []
X = np.array(test['Shell weight']).reshape(-1, 1)
Y_pred = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (mean_squared_error(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE,color = 'r')
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Univariate non-linear regression model")
plt.show()

# 2923 because its the length of training data
x = np.linspace(0, 1, 2923).reshape(-1, 1)

# because p=5 has the lowest rmse
x_poly = PolynomialFeatures(5).fit_transform(x)
reg = LinearRegression()
reg.fit(x_poly, Y_train)
cy = reg.predict(x_poly)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(np.linspace(0, 1, 2923), cy, linewidth=3, color='r')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Curve')
plt.show()

# because the best degree of polynomial is 5 as p=5 has minimum rmse
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate non-linear regression model')
plt.show()


# question 4

print()
print('question - 4')


print('a:')
RMSE = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X_train)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (mean_squared_error(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Multivariate non-linear regression model")
plt.show()


print('b:')
RMSE = []
Y_pred = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X_test)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (mean_squared_error(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))
    # d
    # because the best degree of polynomial is 3 as p=3 has minimum rmse
    if p == 3:
        plt.scatter(Y_test, Y_pred,color = 'r')
        plt.xlabel('Actual Rings')
        plt.ylabel('Predicted Rings')
        plt.title('Multiivariate non-linear regression model')
        plt.show()
        
        
        
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Multivariate non-linear regression model")
plt.show()