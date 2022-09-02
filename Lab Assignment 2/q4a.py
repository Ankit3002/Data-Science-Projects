# Ankit Pal Singh, B20181 , 9149024234
#Question 4
#importing modules
import pandas as pd
from matplotlib import pyplot as plt
import math
# part a(i)

# reading the csv file
landslide = pd.read_csv('landslide_data3_miss.csv')
noisy_df = pd.read_csv('landslide_data3_miss.csv')
# attr is a list which contains the name of all colummns of dataframe
attr = [i for i in landslide.columns]

for i in attr:
    if i !='dates':
        if i !='stationid':
            landslide[i].fillna(int(landslide[i].mean()), inplace=True)
    
# reading original CSV file
original = pd.read_csv('landslide_data3_original.csv')
#orinting the mean
print("The mean of attributes of modified file are")
print(landslide.mean(axis = 0))
print()

# Printing the median 
print("The median of attributes of modified  are")
print(landslide.median(axis = 0))
print()

#Printing the mode 
print("The mode of attributes of modified are")
print(landslide.mode(axis =0))


# Printing the standard deviation
print("The standard deviation of attributes of modified are")
print(landslide.std(axis =0))

# Printing the mean
print("The mean of attributes of original file are")
print(original.mean(axis = 0))
print()

# Printing the median 
print("The median of attributes of original  are")
print(original.median(axis = 0))
print()

#Printing the mode 
print("The mode of attributes of original are")
print(original.mode(axis =0))

print()

# Printing the standard deviation
print("The standard deviation of attributes of original are")
print(original.std(axis =0))

# part a (ii)

# RMSE function takes attribute name as input and gives RMSE value as ouput
def RMSE(a):
    sum = 0
    k = landslide[a] - original[a]
    
    Na = noisy_df[a].isnull().sum()
    for i in k:
        k= float(i)
        sum =sum + math.pow(k,2)
    return math.sqrt((1/Na)*sum)

y_axis = []
x_axis= []
for j in attr:
    if j !='dates':
        if j !='stationid':
            y_axis.append(RMSE(j))
            x_axis.append(j)
            

plt.figure(figsize=(10, 8))     
plt.xlabel('Attribute names')
plt.ylabel('RMSE value ')
plt.title('Bar plot for RMSE value for each attributes')
plt.bar(x_axis,y_axis,width = 0.8)

plt.show()








