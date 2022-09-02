# Ankit Pal Singh , B20181 , 9149024234
#Question 3

# importing modules
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math


# reading the csv files
train_data = pd.read_csv('SteelPlateFaults-train.csv')
test_data = pd.read_csv('SteelPlateFaults-test.csv')

# removing unnamed column which comes because of the indexes
train_data.drop(train_data.columns[train_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
test_data.drop(test_data.columns[test_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)




def bayes_classifier(train,test):
    # calculating the prior probaiblity of each classes 
    prior_class_1 = len(train[train['Class'] ==1]) /len(train)
    prior_class_0 = len(train[train['Class'] ==0]) /len(train)
    
    
    
    # train_class_0 is an dataframe which contains data of class 0
    
    train_class_0 = train[train['Class'] ==0]
    train_class_0 = train_class_0.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    # constant is used to calculate likelihood
    d = len([i for i in train_class_0.columns])
    constant = 1/math.pow(6.28,d/2)
    
    
    # test_0 is used to calculate posteririor probability of every test example with class 0
    test_0 = test.copy()
    test_0 = test_0.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    # cov_0 is the covariance matrix of train data of class 0
    cov_0 = train_class_0.cov()
    # saving the covariance matrix
    cov_0.to_csv('Covariance_0.csv')
    # mean_vector_0 is the mean vector of class 0
    mean_vector_0= train_class_0.mean()
    
    # compare_0 will calculate the posterior probability for class 0
    
    def compare_0(i):
            test_0.iloc[i] = test_0.iloc[i] - mean_vector_0
            inverse_cov_0 = np.linalg.inv(cov_0)
            left = np.dot(test_0.iloc[i].T,inverse_cov_0)
            right = math.exp((-0.5)*(np.dot(left,test_0.iloc[i])))
            likelihood = (constant)* (1/math.sqrt(abs(np.linalg.det(cov_0))))*right
            posterior = likelihood*prior_class_0
            return (posterior)
        
     
    
    # train_class_1 is the dataframe of training data of class 1
    
    train_class_1 = train[train['Class'] ==1]
    train_class_1 = train_class_1.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    # test_1 is used to calculate posteririor probability of every test example with class 1
    test_1 = test.copy()
    test_1 = test.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    # cov_1 is the covariance matrix of train data of class 1
    cov_1 = train_class_1.cov()
    # saving the covariance matrix into csv file
    cov_1.to_csv('Covariance_1.csv')
    
    # mean_vector_1 is the mean of class 1
    mean_vector_1= train_class_1.mean()
    
    # compare_1 will calculate the posterior probability for class 1
    def compare_1(i):
            test_1.iloc[i] = test_1.iloc[i] - mean_vector_1
            inverse_cov_1 = np.linalg.inv(cov_1)
            left = np.dot(test_1.iloc[i].T,inverse_cov_1)
            right = math.exp((-0.5)*(np.dot(left,test_1.iloc[i])))
            likelihood = (constant)* (1/math.sqrt(abs(np.linalg.det(cov_1))))*right
            posterior = likelihood*prior_class_1
            return posterior
    
    predict = []
    for i in range(len(test_1)):
        var_0 = compare_0(i)
        var_1 = compare_1(i)
        
        
        
        if var_0 > var_1:
             predict.append(0)
        elif var_0 < var_1:
             predict.append(1)
        else:
             predict.append(1)
    
    return predict,mean_vector_0,mean_vector_1
    
    
    
# pred is the variable which contains the predicted class output  
pred , mean_0,mean_1 = bayes_classifier(train_data,test_data)
pred = pd.DataFrame(pred)

# real is the variable which contains the  value of class in the  true dataset
real = test_data['Class']

# printing the confusion matrix
print('The confusion matrix is :')
print(confusion_matrix(real,pred))

# printing the accuracy score
print('The accuracy is :')
print(100*accuracy_score(real,pred))

# printing the mean of class 0 and class 1

print(round(mean_0))
print()
print(round(mean_1))