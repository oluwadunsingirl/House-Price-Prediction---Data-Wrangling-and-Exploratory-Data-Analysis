# House-Price-Prediction-Data-Wrangling-and-Exploratory-Data-Analysis
Exploratory data analysis and data wrangling on a housing data, to build a model to predict property price. 
## Introduction
Exploratory data analysis (EDA) and data wrangling on a housing dataset to prepare features for buildig a house
price prediction model.
This project focuses on cleaning the data, handling missing values, exploring relationships and uncovering insights 
of a housing dataset to prepare features for a predictive modeling task.

### Dataset
- source: [kaggle- House Price Prediction Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- shape : 13 columns x 545 rows
### Attribute Info
Features include; price, area, number of bedrooms, guestrooms and bathrooms, stores, proximity to the mainroad, basement,
hot water heating, airconditioning, parking, preferred area and furnishing status.
Target variable : Price

## Tools and libraries
Python (Pandas, NumPy, matplotlib and seaborn)
Baseline Linear Regression model

## Methodology
1. Data Cleaning and Exploratory Data Analysis (EDA):
  - Handling missing values and duplicates.
  - Converted data types where needed.
  - Remove outliers by trimming the bottom and top 10% of properties in terms of price.
  - Removed low and high cardinality features.
  - Drop any columns that would create issues of multicollinearity.
2. Feature Preparation:
  - Encoded categorical variables
  - Normalized numerical features
    
### Result and Evaluation
-Features such as area, number of bedrooms, parking and preferred area showed the strongest relationship with price during
data exploration.

- The linear regression baseline model achieved an r2 score of 0.6529242642153185 meaning that,
the model explains 65% of thevaraince in price.

- The model mean square error was 1.75 trillion, mse shows that the predictions has large error (in millions), meaning the linear model struggles with capturing complex patterns.
- RMSE gives an average prediction error of roughly 1.3 million per house price prediction.

### Conclusion and Recommendation
The baseline linear regression model provides a moderate fit for predicting house prices, explaining about two-thirds of the
variability in the dataset. The high prediction error however suggests that more advanced models such as random forest would be necessary for more accurate predictions.
    
  



