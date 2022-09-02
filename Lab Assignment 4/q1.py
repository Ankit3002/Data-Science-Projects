# Ankit Pal Singh , B20181 , 9149024234
#Question 1

# importing modules


import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# reading the csv file
csv = pd.read_csv('SteelPlateFaults-2class.csv')
# creating a dataframe which have dependent attribute only
y = csv['Class']
# creating a dataframe which have independent attribute only
x = csv.drop('Class',axis = 1)

# splitting the data into training set and testing set
X_train, X_test, X_label_train,X_label_test = train_test_split(x,y,test_size=0.3, random_state=42, shuffle=True,stratify =y)

# train is the dataframe which have both dependent and independent attribute for the training of model
train = X_train.copy()
train['Class'] = X_label_train
train.drop(train.columns[train.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

# test is the dataframe which have both dependent and independent attribute for the testing of model
test = X_test.copy()
test['Class'] = X_label_test
test.drop(test.columns[test.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


# converting the dataframes into csv
train.to_csv('SteelPlateFaults-train.csv')
test.to_csv('SteelPlateFaults-test.csv')

# a and b
acc = {}
for i in [1,3,5]:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,X_label_train)
    X_label_pred = knn.predict(X_test)
    print(f'the confusion matrix when K = {i}:' )
    print(confusion_matrix(X_label_test,X_label_pred))
    print(100*accuracy_score(X_label_test,X_label_pred))
    acc.update({i:accuracy_score(X_label_test,X_label_pred)})
 
    
print('The value of K for which accuracy is high: ') 
print(max(zip(acc.values(), acc.keys()))[1])
    
   



