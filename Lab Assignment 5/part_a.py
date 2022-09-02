# Ankit Pal Singh , B20181 , 9149024234
#Question 3

# importing modules

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

# reading the csv files
train_data = pd.read_csv('SteelPlateFaults-train.csv')
test_data = pd.read_csv('SteelPlateFaults-test.csv')

# removing unnamed column which comes because of the indexes
train_data.drop(train_data.columns[train_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
test_data.drop(test_data.columns[test_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)




def bayes_classifier(Q,train,test):
    # train_class_0 is an dataframe which contains data of class 0
    
    train_class_0 = train[train['Class'] ==0]
    train_class_0 = train_class_0.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    # test_0 is used to calculate log likelihood  of every test example with class 0
    test_0 = test.copy()
    test_0 = test_0.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    
    # compare_0 will calculate the  log likelihood  for class 0
    def compare_0():
        
         gmm = GaussianMixture(n_components = Q ,covariance_type='full')
         gmm.fit(train_class_0)
         score = gmm.score_samples(test_0)
         return score
         
    # train_class_1 is the dataframe of training data of class 1
    
    train_class_1 = train[train['Class'] ==1]
    train_class_1 = train_class_1.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
    
    # test_1 is used to calculate log likelihood of every test example with class 1
    test_1 = test.copy()
    test_1 = test.drop(['Class','X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)
   
    
    # compare_1 will calculate the log likelihood for class 1
    def compare_1():
        gmm = GaussianMixture(n_components = Q , covariance_type='full')
        gmm.fit(train_class_1)
        score= gmm.score_samples(test_1)
        return score
    
    predict = []
    value_0 = compare_0()
    value_1 = compare_1()
    
    for i in range(len(test_1)):
        if value_0[i] > value_1[i]:
            predict.append(0)
        elif value_1[i] > value_0[i]:
            predict.append(1)
            
    return predict
            
            
    
    
acc = {}
for i in [2,4,8,16]:
    
    # pred is the variable which contains the predicted class output  
    pred = bayes_classifier(i, train_data, test_data)
    pred = pd.DataFrame(pred)
    print(f'The confusion matrix for Q = {i} is ')
    # real is the variable which contains the  value of class in the  true dataset
    real = test_data['Class']

    print(confusion_matrix(pred,real))

    # printing the accuracy score
    print('The accuracy is :')
    print(100*accuracy_score(pred,real))
    acc.update({i:accuracy_score(pred,real)})

print('The value of Q for which accuracy is high: ') 
print(max(zip(acc.values(), acc.keys()))[1])




