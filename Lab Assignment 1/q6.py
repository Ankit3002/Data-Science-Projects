# Name = Ankit Pal Singh ,Roll no = B20181
# Question 6

# Importing  modules
import pandas as pd
from matplotlib import pyplot as plt

# reading csv file
csv = pd.read_csv('pima-indians-diabetes.csv')
# removing 'class' column from the dataframe
df = csv.drop('class' , axis = 1)
# Making a list which contains all columns name 
list = []
for i in df.columns:
    list.append(i)
    
# giver function takes the column of dataframe as an argument and returns a list which contains all those values in column
def giver(a):
    x = [i for i in a]
    return x

# printing the boxplot for attributes
for i in list:
    plt.boxplot(giver(df[i]))
    plt.title(f'Boxplot for  {i}')
    plt.show()





