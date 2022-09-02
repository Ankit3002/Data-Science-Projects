# Name = Ankit Pal Singh ,Roll no = B20181
# Question 3
# Importing  modules
import pandas as pd

# reading csv file
csv = pd.read_csv('pima-indians-diabetes.csv')
# removing 'class' column from the dataframe
df = csv.drop('class' , axis = 1)
# Making a list which contains all columns name 
list = []
for i in df.columns:
    list.append(i)

# printing the correlation coefficient between Age and other attributes
for i in list:
        var = df['Age'].corr(df[i])
        print(f'The correlation coffecient between Age and  {i} is {var}')
print()
# printing the correlation coefficient between BMI and other attributes
for i in list:
        var = df['BMI'].corr(df[i])
        print(f'The correlation coffecient between BMI and  {i} is {var}')
        













