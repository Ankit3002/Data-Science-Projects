#Ankit Pal Singh , B20181 , 9149024234
#question 3
# importing modules


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from numpy.linalg import eig
#reading csv file
csv = pd.read_csv('pima-indians-diabetes.csv')

# dropping the class attribute from the dataframe
df = csv.drop('class' , axis = 1)
# making attr which contains all attribute names
attr = [i for i in df.columns]
# df2 is a copy of df dataframe
df2 = df.copy()

# outlier_to_median is a function which replace outliers of df2 with median
def outlier_to_median(a):
    Q1 = df[a].quantile(0.25)
    Q3 = df[a].quantile(0.75)
    IQR = Q3 - Q1
    median = df[a].median()
    df2[a].values[df2[a] > (Q3 + (1.5 * IQR))] = median
    df2[a].values[df2[a] < (Q1 - (1.5 * IQR))] = median
     
for i in attr:
    outlier_to_median(i)


df4 = df2.copy()

# in df4 the standardization is taken place
for a in attr:
    df4[a] = (df4[a] - df4[a].mean())/(df4[a].std())


# pca_df is a function which gives reduced dataframe with dimension n.
def pca_df(df_old,n):
    pca = PCA(n_components = n)
    pca.fit(df_old)
    x_pca = pca.transform(df_old)
    x_pca=pd.DataFrame(x_pca)
    return x_pca

# cov is a function which returns covariance matrix 
def cov(x_pca):
    return np.cov(x_pca.T)

# eigenval is a funciton which calculates eigen values from covariance matrix    
def eigenval(matrix):
    value=eig(matrix)[0]
    return value
# scat plots the scatter plot   
def scat(df):
    plt.scatter(df[0],df[1])
    plt.show()

# pca_df_inverse do the back projection 
def pca_df_inverse(df_old,n):
    pca = PCA(n_components = n)
    pca.fit(df_old)
    x_pca = pca.transform(df_old)
    
    df_final=pd.DataFrame(pca.inverse_transform(x_pca),columns=df.columns)
    return df_final
# error calculates the reconstruction error 
def error(df,n):
    pca = PCA(n_components=n)
    pca.fit(df)
    x_pca = pca.fit_transform(df)
    x_inv = pca.inverse_transform(x_pca)
    df_new=df-x_inv
    df_new=df_new**2
    error=0
    for i, row in df_new.iterrows():
        sum=0
        for j, column in row.iteritems():
            sum=sum+column
        error+=sum**0.5
    return error



#A
a=pca_df(df4, 2)
var=a.var()
print(var)
print(eigenval(cov(a)))
scat(a)

#B
eigenvalue=list(eigenval(cov(df4)))
eigenvalue.sort(reverse=True)
x=[i for i in range(8)]
plt.bar(x,eigenvalue)
plt.xlabel('l value')
plt.ylabel('reconstruction error')


#C
for i in range(1,9):
    print(" reconstruction error for L dim-",i,":",error(df4,i))
    if i>1:
        df_i=pca_df(df4, i)
        
        print(cov(df_i))
        

#D
df_8=pca_df(df4, 8)
print(cov(df_8))
print(cov(df4))




