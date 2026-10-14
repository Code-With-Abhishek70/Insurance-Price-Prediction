# Insurance Price Prediction

A machine learning project that predicts medical insurance charges based on factors like age, BMI, smoking status, number of children and region.

## Overview
The goal is to build a regression model that estimates the insurance cost for a person. The project covers data cleaning, exploratory data analysis, feature engineering, model training and evaluation.

## Dataset
- Source: [insurance.csv]
- Size: 1339 x 7
- Features: age, sex, bmi, children, smoker, region
- Target: `charges`

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Lab

## Workflow
1. Data cleaning: checked missing values and duplicates
2. EDA: analysed how age, BMI and smoking affect charges
3. Feature engineering: encoded categorical columns (sex, smoker, region)
4. Model building: Linear Regression, Random Forest
5. Evaluation: R2 score, RMSE, MAE

## Results
r2  squared val  0.865231697953168
r2 training =  0.8340713711218873
r2 testing =  0.865231697953168

- Best model: Random Forest
- Key insight: smoking status and age have the strongest impact on insurance charges



## Author
Abhishek Kumar || [LinkedIn](https://www.linkedin.com/in/abhishek-kumar-68617a326/)