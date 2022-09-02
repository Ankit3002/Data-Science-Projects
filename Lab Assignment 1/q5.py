# Name - Ankit Pal Singh , Rollno - B20181
# Question 5
# importing modules

import pandas as pd
from matplotlib import pyplot as plt

# reading the csv file
df = pd.read_csv('pima-indians-diabetes.csv')

# group the dataframe according to class
a = df[df['class'] == 0]
b = df[df['class'] == 1]

# giver function takes the column of dataframe as an argument and returns a list which contains all those values in column
def giver(a):
    x = [i for i in a]
    return x

# printing the histogram when attribute class equals 0 
plt.hist(giver(a['pregs']))
plt.title('histogram of pregs for class =0')
plt.xlabel('pregs')
plt.ylabel('frequency of pregs')
plt.show()
# printing the histogram when attribute class equals 1
plt.hist(giver(b['pregs']))
plt.title('histogram of pregs for class =1')
plt.xlabel('pregs')
plt.ylabel('frequency of pregs')
plt.show()
















