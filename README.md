# Telecom-Customer-Churn
Predicting customer churn using machine learning and python

# Business Problem

A teleommunication company (Telco Co.) , who sells residential Voice and Internet services has a massive churn problem as 24% of it's customers churned in the last period.TelCo wants to deploy customer retention strategies by using a predictive model and has contracted with me that meets the following business requireents:

* Find the best prediction model to classify customer churn risk
* Explain the relative influence of each predictor on the model’s predictions
* Suggest potential approaches to reduce customer churn

## Executive Summary
Telco Co. provided select, historical data on 7,043 customers including an indicator as to whether each customer churned. After analyzing and transforming the data, we optimized several classification models. Each model was trained on 70% of the historical data and then tested on remaining 30% test data. Gradient Boosting Model with **84% Recall and 75% AUC is selected as the best model**

### Model Gradient Boost Summary

#### Target Variable: Churn Indicator/Churn Value

  Predictor Features (Strength):

  * (Strong) Contract length: Month-to-month contracts churn much more than 1- or 2-year contracts
  * (Strong) Dependents: Customers supporting dependents (kids/elders etc) churned less
  * (Moderate) Internet services: Internet Fiber or DSL services churned worse than other services
  * (Moderate) Charges : Customers with high monthly charges (>$80)  churns more comparitively to average monthly charges(<$60)
  * (Moderate) Security: Customers not availing Security Services are 3 times more probable to churn
  * (Moderate) Tenure Months : Customers with Tenure less than 12 months has higher chance to leave the company
  * (Low) Streams: Customer using streaming of TV or Movies on Internet service have greater probabilty to churn


#### Gradient Boosting Key Metrics

* 53.6% improvement over baseline model
* 84% Recall means model predicts 84% of customer churn cases correctly
* 75% AUC indicates model is correct on nearly 3 out of 4 customers cases
* Using a probablility of churning threshold at 0.5, the business will be able to save $135K/month savings over no model in place, and will be able to save $66K/month over spending money on retention effort for all customers 
* This model will be socialized with Telco Co. and then refined based on feedback before moving towards deploying this into the business.


# Introduction

Low switching costs for customers (supported by government regulations) mean that customer loyalty is the only real tool that telecom companies must have to reduce their churn rates.  Connected data, used to improve service quality, dynamically adjust pricing/promotions, and offer personalized content to consumers, enable telecom providers to influence customer loyalty and increase customer retention directly.A machine learning algorithm on historical data will help the Telco to understand the customers well. It will address various questions such as :
1. Which customers are more probable to churn
2. What are the payment habits for churning customers?
3. How does customer lifespan matters in terms of identifying churns? etc

# Exploratory Data Analysis with Vizualization
Here in this [notebook]() I do basic exploratory data analysis on the dataset to get an understanding of the data. Python packages like matplotlib and seaborn are used. Things covered :

* Getting an understanding of the data
* Checking missing values and treating them 
* Analazing  the numerical feature distribution
* Analysis all features with respect to churn
* Performing K-means clustering to understand churning behaviour
* In depth visualizations to have good understanding of the customer base

## Data Findings

### Class Imbalance : Customer Churning Rate: 24%
There is a class imbalance , ie distribution of records across the classes( Churn/Non-Churned) is not equal. Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. This results in models that have poor predictive performance,specifically for the minority class (Churn)

![image](https://user-images.githubusercontent.com/49127037/140165675-6111c524-e74a-4180-a299-00dff7933425.png)

**1. Tenure and Churning**- A large proportion of the customers either have a very short duration (< 3 months) or a quite long duration ( >5 years).
Churning customers have much lower tenure with a median of 10 months compared to a median of 38 months for non churners. Business can pay attention to customers who are about to complete 10 months
![image](https://user-images.githubusercontent.com/49127037/139969885-ceca3cc2-cebe-4e6b-a223-d1505a37eaf0.png)
We also see that about 60% of churning happens in the 1st month of the customer tenure, and post 5 months churning starts to stable and cuts down to just 35%. 
![image](https://user-images.githubusercontent.com/49127037/139970514-e2240abb-98b4-4ad1-b142-723f913ea091.png)

Also, The churn rate is highest, almost 50% for the customers with less than a year of tenure
![image](https://user-images.githubusercontent.com/49127037/140003057-9608571b-c950-4132-a47b-ef1b860f1e69.png)

There can be multiple reasons for this.Perhaps the joining offers given to a customer is just given for the first month and after that connection price becomes so high that customer tends to leave. In this case business can increase the duration of the  joining offer at least for 3-6 months and monitor the customer's usage and based on that offer them special offers.

**2. Monthly Charges and Churning**- Majority of the Churning customers have higher monthly charges with a median of $80. For non-churners, it is within $60. Business can keep a tab on the customers who are paying higher monthly charge and promote special bundle offers helping them curb these charges
![image](https://user-images.githubusercontent.com/49127037/139970272-fc93326f-8374-4f38-bca4-bcedd4d27b66.png)

**3.Tenure and Monthly Charge** - A k- means clustering on Tenure and Monthly charges for churned customers show that  a significant numer of relatively new customers but those who have subscribed to more services ( having higer monthly charges) are the ones who are more likely to leave. Business can try to offer a promotional subsidized extra services plan, so that new customers can enjoy the services by paying cheaper and not leave the company

![image](https://user-images.githubusercontent.com/49127037/140007278-adf8d3fa-1c6c-4cb3-8ad7-8632d03d0513.png)


**3. Senior/Non-Senior citizens and Churn Rate** - Churn rate is higher for senior citizen customers. Since, the share of senior citizens is about 16% from the total amount of clients, this indicator requires further research with additional data. This indicator is significant and should be taken into account when creating a loyalty program
![image](https://user-images.githubusercontent.com/49127037/139981788-95ce0335-27ec-4f7e-926f-b50028177531.png)

**4. Parter, Dependents and Churn Rate** - Having no Partner or dependents, definitely increases the probability to churn
![image](https://user-images.githubusercontent.com/49127037/139983901-42a06852-ed00-4b9e-9bf0-7f97baff446a.png)

**5. Internet Service and Churn Rate** - Clients with Fiber optic internet service is _**2.2**_ times more probable to leave the company than clients with DSL Internet service. The share of clients with Fiber optic internet comprises  44% of the clients base, which is quite significant. Also, an interesting fact is, customers who does not avail any internet services are churning the least .

Several reasons are possible: may be, lower cost substitutes from competitors or the service is not smooth. Business can target these customers with feedback mails.
![image](https://user-images.githubusercontent.com/49127037/139998616-7533b479-f3f3-4a98-a763-cba0f482867c.png)

**6. Online Security ,Tech Support and Churn Rate** - Almost 50% of the Clients do not avail Online security and Tech support, and these customers are almost 3 times more probable to leave, compared to customers who availing these services. Business can target these customers by offering cheap options of security plans and time to time customer support
![image](https://user-images.githubusercontent.com/49127037/139999687-9c4b7b4f-3cea-4d8f-903f-d4abdd6ad926.png)

**7. Online Backup,Device Protection and Churn Rate** - Almost 45% customers prefer no Online Backup and Device protection services, and the churn rate is _**1.7 times**_ more than the clients who prefers these services. The business can create programs where they can offer these customers some cheap promotion offers and then monitor their behaviour
![image](https://user-images.githubusercontent.com/49127037/140000119-1b073d80-f113-4675-8be4-a2f00621eb38.png)

**8. Contract Type and Churn Rate** - A majority, almost 55% of the clients have opted Month-To-Month Contract type, but 43% has left the company. This churn rate is 14.3 times more than clients with 2-Year Contract and 4 times more than clients with 1-Yr Contract. This is quite a flag for the business. Without any contract holding them back, customers can easily change mobile operators. This means they have little loyalty to brands, and bad experiences can spread more rapidly. Business can target these month-to-month contractors with some subsidized cost Year-contracts and check the conversion rates
![image](https://user-images.githubusercontent.com/49127037/140001033-aa47bc58-4b8a-45ea-9fd6-7ef09b326ebb.png)

It is seen, with Month-to-month contract, for the majority of the clients, the tenure duration is quite low. But if the contract type is 2 years, majority of the customers have a longer tenure.
![image](https://user-images.githubusercontent.com/49127037/140002409-ec4800e3-fcdc-49f2-bd72-bfd1062a54e0.png)


**9. Paperless Billing** - Almost 60% of the customers prefers paperless billing, however, the churning rate is 2 times more than customers who prefer Paper Billing. Business can provide Paperless Billing customers with the option of converting to Paper Billings and get feedback
![image](https://user-images.githubusercontent.com/49127037/140001875-88d7da37-006e-4413-aa46-ccc270fe8610.png)


**10. Payment method** - 30% of the client base has opted for payment in Electronic Check payments, however almost 45% of them has churned.  Electronic-Check Payment customers can be offered with options of automatic payment modes such as Credit Card payments or Bank Transfer which has much lower churning rate
![image](https://user-images.githubusercontent.com/49127037/140001972-a87c3f42-d91f-4c68-97ff-e912564bc51c.png)

**11. Customers availing multiple services and churning** - The figure shows that customers availing just one service has greater percentage of churning compared to customers availing all 6.They might be hesitant to cancel a contract, when they depend on the additional service components (e.g. security ,backup etc).

![image](https://user-images.githubusercontent.com/49127037/140006164-889bd430-eaf5-412d-9923-d1c9f7a55cb7.png)

**12 . Services which has the highest churners** - Through this figure, we see that clients opting for Streaming Tv and Movies, see the highest churning. With more data, we can analyse how much extra they are paying for these services, and if these extra cost is driving them to leave the company


![image](https://user-images.githubusercontent.com/49127037/140006194-8c21d1ac-cc2c-449c-8fed-49814ce3efcb.png)

# Feature Engineering and Data Preprocessing
In this [notebook]() I have tried following things :

* Created few features with the help of existing features , so as to derive more information to feed into the model . I have also created few graphs for data visualization to see how this new features are defining the churning rate
* Created dummy variables for the categorical features.
* A correlation figure between churn and other features is produced to understand which features are highly correlated

## Features used for the model
The important features which are used in building models are Tenure Hroup, Type of Internet Service, Customer's payment type etc.  As we have seen from the data exploration and the correlation graph, these features have significant importance in predicting the probability of customer leaving the company

![image](https://user-images.githubusercontent.com/49127037/140008015-720c700f-037d-4eaa-9244-a22c903c3338.png)


# Training models,evaluation and hyperparametertuning
Built models and then evaluate them in this [notebook](). Steps I have done are :

* Dividing the data into two parts: training and test. To find the model, I used the training set and finally test the model on unseen data, which is the test set
* Because of class imbalance, creating 3 samples using resampling techniques
* Creating a baseline model using the Logistic regression on the original sample. Recall and Area under the curve (AUC) is the evaluation metric. Other complex models will be     tested against the baseline model
* 3 models are fitted and one with balance of highest recall and AUC is chosen as the final model
* Hyperparameter tuning of the best models and finally testing on the unseen data
* Getting the feature importance of the model which shows, which features have contributed the most to the final model

## Evaluation metric
* Recall
* ROC/AUC


## Models
**1. Baseline model** : A Logistic Regression model using Logistic Regession on 4 samples 

Sample | Recall
------------ | -------------
Original Sample | 0.56
Resample using Upsampling minority class | 0.80
Resample using SMOTE | 0.83
Resample using Downsampling majority class | 0.80

So without any model, the **baseline model has a recall = 0.56**. Using resampling technique using SMOTE, 3 models (Logistic Regression,Random Forest Classifier, Gradient Boosting Classifier are fit to improve the recall and also optimizing the Area under the Curve ( AUC)

**2. Final Model** : **Gradient Boosting is choosen as the final model, based on Recall score and AUC**

![image](https://user-images.githubusercontent.com/49127037/140026315-4eb20389-9151-4cc3-9e63-1337b0e6c854.png)
**Gradient Boosting Confusion Matrix :**

![image](https://user-images.githubusercontent.com/49127037/140027442-770fd38a-f759-462a-a7ab-4cff348a847c.png)

**3. Feature Importance** - According to the model, follwoing are the key features in determinig customer churn:

- Contract length: Month-to-month contracts churn much more than 1- or 2-year contracts
- Dependents: Customers supporting dependents (kids/elders etc) churned less
- Internet services: Internet Fiber or DSL services churned worse than other services
- Charges : Customers with high monthly charges (>$80)  churns more comparitively to average monthly charges(<$60)
- Security: Customers not availing Security Services are 3 times more probable to churn
- Tenure Months : Customers with Tenure less than 12 months has higher chance to leave the company
- Streams: Customer using streaming of TV or Movies on Internet service have greater probabilty to churn
- 
![image](https://user-images.githubusercontent.com/49127037/140232778-53081b8b-6006-485a-98f8-4c339f2fcf6e.png)

## Cost Evaluation
Cost evaluation is explored to  understand the cost implications after implementing the model.  Following cost are attributed on the results of confusion matrix.

* **TN : USD 0** - The model correctly identified a loyal customer, in this case, business does not bear any action cost
* **FN : USD 500** - As we have discussed earlier, False negatives have grave implications because acquiring a replacement customer and all other associated costs are huge. Business approximately pay, example 500 for each customer where the model incorrectly predicts that a customer will stay
* **TP and FP : USD 100** -  Model predicts these customers as churning, so business puts in a retention cost such as setting up few promotions, ad cost etc of an amount of say 100 USD per customer . Three scenarios are soughted and following is the Cost to Company and savings.

![image](https://user-images.githubusercontent.com/49127037/140232268-48a6a46b-5ab6-411a-98b4-d5c2f2729961.png)

Using a probablility of churning threshold at 0.5, the business will be able to save $135K/month savings over no model in place, and will be able to save $66K/month over spending money on retention effort for all customers . If lower threshhold, more optimization on cost savings

## Next Steps

Telcos typically have much more data available that could be included in the analysis, like extended customer and transaction data from CRM systems and operational data around network services provided. Also they typically have much larger amounts of churn/non-churn events at their disposal than the ca. 7000 in this case example. With those, neural networks could be properly trained to detect more complex patterns in data and achieve higher accuracies. A high accuracy is needed to be able to identify promising customer cases where churn can be avoided as, eventually, the customer returns protected need to outweigh the costs of related retention campaigns.































