# B20181  , Ankit Pal Singh 
# mobile number -- 9149024234


# importing the modules
import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


#reading csv file
csv = pd.read_csv('Iris.csv')

#pca_maker gives reduced dataframe with dimension n.
def pca_maker(df , n):
    pca = PCA(n_components = n)
    pca.fit(df)
    x_pca = pca.transform(df)
    x_pca=pd.DataFrame(x_pca)
    return x_pca

train = csv.drop('Species' , axis = 1)


# Question 1
# computing the covariance matrix.
covariance_mat = np.cov(train.T)

val = eig(covariance_mat)

# plotting the eigen values 
plt.plot(val[0])
plt.xlabel('eigen value is at integers only')
plt.ylabel('eigen values')
plt.show()


train_reduced = pca_maker(train,2)

# Question 2

def k_means(train_data , k):
  
    model = KMeans(n_clusters=k)
    model.fit(train_data)


    predict = model.predict(train_data)
    return predict , model


label , kmeans = k_means(train_reduced, 3)

u_labels = np.unique(label)
#Getting the Centroids
centroids = kmeans.cluster_centers_
    



for i in u_labels:
    filtered_label = train_reduced[label == i]
    x=filtered_label.loc[:,0]
    y = filtered_label.loc[:,1]
    plt.scatter(x,y ,label = label)

#part_a    
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend(u_labels)
plt.show()

# part_b
print(f'The distortion measure is {kmeans.inertia_}')

#part_c

def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 #print(contingency_matrix)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)


    


purity = purity_score(csv['Species'], label)

print(f'The purity score when k = 3 is : {purity}')


# Question 3

clusters = [2,3,4,5,6,7]
purity_list = []   # purity_list contains purity scores with given value of clusters
j = []  #  j contains distortion measures with given value of clusters
for i in clusters:
    label , kmeans = k_means(train_reduced,i)
    j.append(kmeans.inertia_)
    purity_value = purity_score(csv['Species'] , label)
    print(f'The value of purity score for k = {i} is {purity_value}')
    purity_list.append(purity_value)
    
    
plt.plot(clusters, j)
plt.xlabel('value of clusters(k)')
plt.ylabel('value of distortion measure')
plt.show()


















