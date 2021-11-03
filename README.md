# Telecom-Customer-Churn
Predicting customer churn using machine learning and python

# Business Problem

# How can a telecom industry identify customers who are churning and stop them?

Customer loyalty is the key to profitability in the telecom industry. A telecom business sector can use this model to early identify customers who are most likely to leave their service and take preventive measures in order to retain them. Promotions and various discounted offers are often send to customers, however, customers who are probable churn require special attention. This machine learning model takes into different into different factors such as  demographic information , nilling services etc,analyzes all these factors throughly with  visualization and statistical models, and thereby solves a huge problem of the telecom business.


# Introduction

Low switching costs for customers (supported by government regulations) mean that customer loyalty is the only real tool that telecom companies must have to reduce their churn rates.  Connected data, used to improve service quality, dynamically adjust pricing/promotions, and offer personalized content to consumers, enable telecom providers to influence customer loyalty and increase customer retention directly.A machine learning algorithm on historical data will help the business to understand the customers well. It can address various questions such as :
1. Which customers are more probablo churn
2. What are the payment habits for churning customers
3. Whether ...
4. 

# Data Collection
The dataset is downloaded from IBM datasets on telecom data. It contains around 7000 customer records with 21 features such as demograhic features : gender, do the partner or depents, service related- how many services they use, what types of services they use ad billing information : whaht is their paymont mode, what is their contract type etc. . The machine learning model will be able to identify future customers who are probable to churn

# Exploratory Data Analysis with Vizualization
Here in this [notebook]() I do basic exploratory data analysis on the dataset to get an understanding of the data. Python packages like matplotlib and seaborn are used. Things I covered :

* Getting an understanding of the data
* Checking missing values and treating them 
* Analazing  the numerical feature distribution
* Analysis all features with respect to churn
* In depth visualizations to have good understanding of the customer base

## Data Findings
**1. Tenure and Churning**- A large proportion of the customers either have a very short duration (< 3 months) or a quite long duration ( >5 years).
Churning customers have much lower tenure with a median of 10 months compared to a median of 38 months for non churners. Business can pay attention to customers who are about to complete 10 months
![image](https://user-images.githubusercontent.com/49127037/139969885-ceca3cc2-cebe-4e6b-a223-d1505a37eaf0.png)

**2. Monthly Charges and Churning**- Majority of the Churning customers have higher monthly charges with a median of $80. For non-churners, it is within $60. Business can keep a tab on the customers who are paying higher monthly charge and promote special bundle offers helping them curb these charges
![image](https://user-images.githubusercontent.com/49127037/139970272-fc93326f-8374-4f38-bca4-bcedd4d27b66.png)


































