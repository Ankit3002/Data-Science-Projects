# Ankit Pal Singh, B20181 , 9149024234
#Question 2
#importing modules
import pandas as pd
from matplotlib import pyplot as plt

# part a

# reading the csv file
landslide = pd.read_csv('landslide_data3_miss.csv')

# dropping the tuples which have missing values in the stationid attribute
df = landslide.dropna( how='any',subset=['stationid'])

print('The total numbers of tuples deleted are:',len(landslide)-len(df))
print()
# part b

# removing those rows which have total nan values greater or equal to 1/3 of (no. of attributes)

newDF=df.dropna(how='any',axis='rows',thresh=7)
#printing the total no. of tuples which are dropped
var = len(df)-len(newDF)    
print('the total no. of tuples which are dropped because of 1/3 criteria',var)
    
print()
#Question 3

# column_nan is a panda series which contain number of nan values of each attribute
column_nan = newDF.isnull().sum()
#printing the number of missing values in each attributes
print('the number of missing values in each attributes')
print(column_nan)
print()
#printing the total number of missing values in the file
print('the total number of missing values in the file is ',column_nan.sum())
    
    
    
    
    
    
    
    
    