# Ankit Pal Singh, B20181 , 9149024234
#Question 1
#importing modules
import pandas as pd
from matplotlib import pyplot as plt



# reading the csv file
landslide = pd.read_csv('landslide_data3_miss.csv')

# attr is a list which contains name of all colummns of dataframe
attr = [i for i in landslide.columns]

# function which takes input as attribute and output as frequency of that attribute

def get_frequency(a):
    df = landslide.isnull()
    return len(df[df[a] == True])

# creating a list which contains freqency of nan values of each attribute
freq = [get_frequency(i) for i in attr]

# creating the bar plot
plt.figure(figsize=(10, 8))

plt.bar(attr,freq, width = 0.8)
plt.xlabel('Attribute names')
plt.ylabel('Number of missing values')
plt.title('Bar plot for the Number of missing values for each attributes')
plt.show()
