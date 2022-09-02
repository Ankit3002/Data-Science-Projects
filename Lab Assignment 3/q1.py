#Ankit Pal Singh , B20181 , 9149024234
#question 1
# importing modules

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#reading csv file
csv = pd.read_csv('pima-indians-diabetes.csv')

# dropping the class attribute from the dataframe
df = csv.drop('class' , axis = 1)
# df2 is a copy of df dataframe
df2 = df.copy()
# making attr which contains all attribute names
attr = [i for i in df.columns]

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

# making df3 and df4 to do normalization
df3 = df2.copy()
df4 = df2.copy()
# in df3 the min max normalization is taken place
# in df4 the standardization is taken place
for a in attr:
    df3[a] = ((df3[a] - df3[a].min())/(df3[a].max()-df3[a].min()))*(7) + 5

    df4[a] = (df4[a] - df4[a].mean())/(df4[a].std())
print("The minimum of values before performing min max normalization")
print(df2.min())
print("The maximum of values before performing min max normalization")
print(df2.max())

print("The minimum of values after performing min max normalization")
print(df3.min())
print("The maximum of values after performing min max normalization")
print(df3.max())

print("Mean before doing standardization")
print(df2.mean(axis = 0))
print("standard deviation before doing standardization")
print(df2.std(axis = 0))

print("Mean after doing standardization")
print(round(df4.mean(axis = 0)))
print("standard deviation after doing standardization")
print(df4.std(axis =0))


    