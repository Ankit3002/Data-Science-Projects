#Ankit Pal Singh , B20181 , 9149024234
#question 5
# importing modules
import pandas as pd
import matplotlib.pyplot as plt
#reading csv file
landslide = pd.read_csv("landslide_data3_miss.csv") 

# interpolating the data
landslide1 = landslide.interpolate()
# plotting the boxplot for temperature
plt.figure()
landslide1.boxplot(['temperature'])

# finding the percentiles and IQR
Q1 = landslide1['temperature'].quantile(0.25)
Q3 = landslide1['temperature'].quantile(0.75)
IQR = Q3 - Q1
# plotting the boxplot for rain
plt.figure()
landslide1.boxplot(['rain'])

b1 = landslide1.loc[(landslide1['temperature'] > (Q3 + (1.5 * IQR))) | (landslide1['temperature'] < (Q1 - (1.5 * IQR)))]
print(b1['temperature'])

Q1b = landslide1['rain'].quantile(0.25)
Q3b = landslide1['rain'].quantile(0.75)
IQR2 = Q3b - Q1b

print(Q1b)
B2 = landslide1.loc[(landslide1['rain'] > (Q3b + (1.5 * IQR2))) | (landslide1['rain'] < (Q1b - (1.5 * IQR2)))]
print(B2['rain'])

#b
Median1 = landslide1['temperature'].median()
landslide2 = landslide1.replace({'temperature' : {7.6729 : Median1}})
plt.figure()
landslide2.boxplot(['temperature'])

Median2 = landslide1['rain'].median()
landslide1['rain'].values[landslide1['rain'] > (Q3b + (1.5 * IQR2))] = Median2
landslide1['rain'].values[landslide1['rain'] < (Q1b - (1.5 * IQR2))] = Median2
plt.figure()
landslide1.boxplot(['rain'])

