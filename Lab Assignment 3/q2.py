#Ankit Pal Singh , B20181 , 9149024234
#question 2
#importing modules


from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import eig
from sklearn.decomposition import PCA
import pandas as pd


# plot is a function which takes dataframe as a input and print a scatter plot as an output.
def plot(df):
    plt.scatter(df[0],df[1])
    plt.show()

#plot_eigendirection takes dataframe as input and plot the eigen vectors over scatter plot
def plot_eigendirection(df):
    plt.scatter(df[0],df[1])
    plt.quiver(0,0,eigenvectors[0][0],eigenvectors[0][1],color="Red")
    plt.quiver(0,0,eigenvectors[1][0],eigenvectors[1][1],color="Red")

#approximation find the projection of data on eigen directions    
def approximation(df,v):
    d=df-df.mean()
    for i in range(len(d)):
         d[i]=np.matmul(d[i],np.transpose(v))
         
    approx=[]
    for i in range(len(d)):
        approx.append([d[i][0]*v[0],d[i][1]*v[1]])    
    approx=pd.DataFrame(approx)
    return approx 

# error function calculates reconstruction error using euclidian distance
def error(df):
    pca = PCA(n_components=2)
    pca.fit(df)
    x_pca = pca.fit_transform(df)
    x_inv = pca.inverse_transform(x_pca)
    error=0
    for i in range(len(df)):
        error+=((df[i][0]-x_inv[i][0])*2)+((df[i][1]-x_inv[i][1])*2)*0.5
    return error



# A
mean = [0,0]
cov = [[13,-3],[-3,5]]
D = np.random.multivariate_normal(mean, cov, 1000)


eigenvalue = eig(cov)[0]
eigenvectors = eig(cov)[1]

df = pd.DataFrame(D)
plot(df)

#B

print('The eigenvalues are ',eigenvalue)
print('The eigenvectors are ',eigenvectors)
plot_eigendirection(df)
plt.show()

#C
e1 = approximation(D,eigenvectors[0])
e2 = approximation(D,eigenvectors[1])
plot_eigendirection(df)
plot(e1)

plot_eigendirection(df)
plot(e2)

# D
print("The reconstruction error is :")


print(float("{0:.3f}".format(error(D))))


