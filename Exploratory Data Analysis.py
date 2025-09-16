# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# read csv file
uploaded = files.upload()
df = pd.read_csv("Housing.csv")
df.head()

#check for number of rows and colunmns
df.shape

#get summary data
df.info()
df.describe()
# check for null values
df.isna().sum()   

# check for duplicates
df.duplicated().sum()

# checking for the number of unique values
df.select_dtypes('object').nunique()

# handling categorical data with yes or no columns values

df['mainroad'] = df['mainroad'].str.strip().str.lower().map({'yes':1,'no':0})

df['guestroom'] = df['guestroom'].str.strip().str.lower().map({'yes':1,'no':0})

df['basement'] = df['basement'].str.strip().str.lower().map({'yes':1,'no':0})

df['hotwaterheating'] = df['hotwaterheating'].str.strip().str.lower().map({'yes':1,'no':0})

df['airconditioning'] = df['airconditioning'].str.strip().str.lower().map({'yes':1,'no':0})

df['prefarea'] = df['prefarea'].str.strip().str.lower().map({'yes':1,'no':0})

#encoding the values in the furniture status column
df = pd.get_dummies(df, columns =['furnishingstatus'], drop_first = True)

#check cor correlation between features and target
df.corr()

# using a barchart to display the correlation btw features
corr = df.corr()['price'].drop('price')
plt.figure(figsize=(8,5))
sns.barplot(x=corr.values, y = corr.index)
plt.show()

#more EDA to understand the pattern in the data
plt.hist(df['price'])
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#a scatterplot to show the relationship between price and area
plt.scatter(x = df['price'], y = df['area'])
plt.show()  










