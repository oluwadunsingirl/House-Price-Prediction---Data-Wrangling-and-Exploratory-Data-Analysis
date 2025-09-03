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

#split data 80% for training and 20% for testing the model
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#A baseline model building
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('r2 score:', r2_score(y_test, y_pred))
print('mse:', mean_squared_error(y_test, y_pred))

## r2 score = 0.6529242642153185
mse: 1754318687330.6633







