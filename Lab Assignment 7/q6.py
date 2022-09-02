# B20181  , Ankit Pal Singh 
# mobile number -- 9149024234



# Question 6


# importing the modules
import pandas as pd
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN



#reading csv file
csv = pd.read_csv('Iris.csv')

#pca_maker converts gives reduced dataframe with dimension n.
def pca_maker(df , n):
    pca = PCA(n_components = n)
    pca.fit(df)
    x_pca = pca.transform(df)
    x_pca=pd.DataFrame(x_pca)
    return x_pca

train = csv.drop('Species' , axis = 1)


train_reduced = pca_maker(train,2)

def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 #print(contingency_matrix)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)


def model_label_dbscan(epsilon , min_points , train_data):
    dbscan_model=DBSCAN(eps=epsilon, min_samples= min_points).fit(train_data)
    DBSCAN_predictions = dbscan_model.labels_
    return dbscan_model , DBSCAN_predictions
    
eps_list = [1,1,4,4]
min_points_list  = [5,10,5,10]
for i,j in zip(eps_list , min_points_list):
    model,label = model_label_dbscan(i,j, train_reduced)
    centroids = model.components_
    
    

    u_labels = np.unique(label)
    for k in u_labels:
        filtered_label = train_reduced[label ==k]
        x=filtered_label.loc[:,0]
        y = filtered_label.loc[:,1]
        plt.scatter(x,y)
    plt.legend(u_labels)
    plt.title(f'Scatter plot for epsilon = {i} and min_samples = {j}')
    plt.show()
    purity_value = purity_score(csv['Species'] , label)
    print(f'The value of purity score for eps = {i}  and min_samples = {j} is {purity_value}')
    
        


















