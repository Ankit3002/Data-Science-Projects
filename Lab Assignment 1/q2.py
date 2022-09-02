# Name = Ankit Pal Singh ,Roll no = B20181
# Question 2
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
# giver function takes the column of dataframe as an argument and returns the list
def giver(a):
    x = [i for i in a]
    return x
# printing scatter plot between age and all other attributes excluding class
for i in list:
    if i !='Age':
        plt.scatter(giver(df['Age']),giver(df[i]))
        plt.title(f'Scatter plot between Age and {i}')
        plt.xlabel('Age')
        plt.ylabel(i)
        plt.show()
# printing scatter plot between BMI and all other attributes excluding class
for i in list:
    if i !='BMI':
        plt.scatter(giver(df['BMI']),giver(df[i]))
        plt.title(f'Scatter plot between BMI and {i}')
        plt.xlabel('BMI')
        plt.ylabel(i)
        plt.show()













