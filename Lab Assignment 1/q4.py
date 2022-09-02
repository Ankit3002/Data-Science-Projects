# Name - Ankit Pal Singh , Rollno - B20181
# Question 4
# importing modules

import pandas as pd
from matplotlib import pyplot as plt

# reading the csv file
df = pd.read_csv('pima-indians-diabetes.csv')

# giver function takes the column of dataframe as an argument and returns a list which contains all those values in column
def giver(a):
    x = [i for i in a]
    return x

# printing the histogram 
plt.hist(giver(df['pregs']))
plt.title('Histogram for Number of times pregnant')
plt.xlabel('pregs')
plt.ylabel('frequency of pregs')
plt.show()
plt.hist(giver(df['skin']))
plt.title('Histogram for Triceps skin fold thickness ')
plt.xlabel('Skin')
plt.ylabel('frequency of skin')
plt.show()
        























