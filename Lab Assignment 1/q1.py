# Name = Ankit Pal Singh ,Roll no = B20181
# Question 1
# Importing  modules
import pandas as pd

# Reading CSV file
csv = pd.read_csv('pima-indians-diabetes.csv')

# removing 'class' column from the dataframe

df = csv.drop('class' , axis =1)

# Printing the mean 
print("The mean of attributes are")
print(df.mean(axis = 0))
print()

# Printing the median 
print("The median of attributes are")
print(df.median(axis = 0))
print()

#Printing the mode 
print("The mode of attributes are")
print(df.mode(axis =0))
print()

# Printing the minimum 
print("The minimum of attributes are")
print(df.min())
print()

# Printing the maximum
print("The maximum of attributes are")
print(df.max())
print()

# Printing the standard deviation
print("The standard deviation of attributes are")
print(df.std(axis =0))




















