
### Context:
Any business wants to maximize the number of customers. To achieve this goal, it is important not only to try to attract new ones, but also to retain existing ones. Retaining a client will cost the company less than attracting a new one. In addition, a new client may be weakly interested in business services and it will be difficult to work with him, while old clients already have the necessary data on interaction with the service.

Accordingly, predicting the churn, we can react in time and try to keep the client who wants to leave. Based on the data about the services that the client uses, we can make him a special offer, trying to change his decision to leave the operator. Thanks to this, the task of retention will be easier to implement than the task of attracting new users, about which we do not know anything yet. 

### Objective:

We are provided with a dataset from a telecommunications company. The data contains information about almost six thousand users, their demographic characteristics, the services they use, the duration of using the operator's services, the method of payment, and the amount of payment. The task is to analyze the data and predict the churn of users (to identify people who will and will not renew their contract) by doing Exploratory Data Analysis and building Machine Learning prediction models. We investigate the following tasks in this report:

• Exploratory Data Analysis.

• Research of dependencies and formulation of hypotheses.

• Is there any bias in the dataset? How can it be addressed?

• Building prediction models and compare the quality of the obtained models?

• How can we improve techniques previously proposed?

### 1. Exploratory Data Analysis:
Before going to analysis, we need to look at the data for null values and categorical variables. So, I performed a lot of pre-processing steps, but I am not mentioning them here as this report is more on explaining the above-mentioned business objectives. If you want to know more on data pre-processing steps, please refer to the code [here](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Code.ipynb) as I have explained everything more clearly.

#### 1.1.	Initial Observations:

•	Only 16% of the population are Senior Citizens and only 30% of the population have dependents.

•	10% of the population still does not have phone service. But our aim here is to retain the existing customers, not to get new customers. So, we do not focus more on this.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture1.jpg)

•	44% population have Fiber Optic, 35% have DSL and 21% have no Internet connection.\

•	Only 29% have Online Security and Tech Support, 35% have online Backup & Device Protection, 39% have Streaming TV & Streaming Movies.

•	More than half population have Month-to-month contract, while 21% have one-year contract, 24% have two-year contract. 

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture2.jpg)

•	Almost 60% of the population are doing paper less billing.

•	In payment methods, people are paying more through Electronic check (34%), while the remaining three modes are equally distributed.

#### 1.2.	Correlations:

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture3.jpg)

So, there are some obvious correlations here. For example, if a person has a higher tenure (meaning they have been a client for longer), then they will have more total charges. So, they have 0.83 correlation. Monthly charges are also clearly correlated with Total Charges. It also seems that tenure is most related to contract with a 0.67 correlation, both of which are most closely related to Churn as well. 

What is interesting though is that none of these columns are really that good at determining the churn. One would expect that if someone has been a client for a long time (tenure), they would stay a client, but that does not seem to be the case with a -0.35 correlation. Flipping a coin would be a better determination of if someone is staying or not. Same thing with monthly and total charges, there seems to be very little correlation between them and churn which is, again, surprising because how much someone is being charged would seem like a great predictor of whether or not they will renew their subscription but it doesn't seem so.

We can see an obvious correlation between Streaming TV and Streaming Movies; however, from the below figure “it doesn't seem like people do not significantly renew their contract even if they stream TV and/or Movies.”

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture4.jpg)
 
It also doesn't seem like any of the securities (Online Security, Online Backup and Device Protection) are correlated with each other meaning that “having one type of security like Online Security has almost no effect on whether someone will have another type of security like Device Protection.” There seems to be no significant correlation between the security someone has and them renewing their contract with Telecom. Security is not that helpful of a predictor when analyzing the churn. Perhaps this correlation chart is not giving us the full picture, lets analyze the data a bit more.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture5.jpg)
 
So, it turns out that the Pearson correlation does not do the features justice as we can clearly see a much clearer correlation with Churn now. If Total Charges or Monthly Charges are small, people are also much more likely to keep their contract with telecom. 
“An interesting thing to note here is that even if monthly charges are higher, people with a high tenure (have stayed with telecom for a while) will most likely still renew their contract.”

The type of Contract someone has with company seems to play a significant role in whether someone will renew their contract or not. Having a 2 year or 1-year contract means that people will most likely renew their contract while most people with a month to month contract tend to not renew their contracts. 

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture6.jpg)
![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture7.jpg)

“What's interesting to notice is that for two year and one year contracts, the median monthly payment can be fairly  high (about 100) before people start becoming less likely to renew their contract, however, the month to month charge does not need to be as high before people stop renewing their contracts.”
In fact, the median total charges for people who did not renew their month to month contract is smaller than the median total charges for people who did renew their month to month contract.


#### 1.3.	Bias in dataset:

As you can see from the below graphs there seems to be no gender bias and no bias in marriage of the population.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture8.jpg)
![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture9.jpg)
 
### 2.	Model Building:

I am combining both the datasets that they gave and splitting it into train (80%) and test (20%). I am using KNN (as it is computationally fast) and XG boost (as it is ensemble technique and gives better accuracy) classifiers for fitting the train dataset and predicting using test dataset. I am using ‘Randomized Search Cross Validation’ for hyper parameter tuning in XG boost. We will look at the evaluation metrics for comparing the models.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Result1.PNG)

As you can clearly see that XG-boost is better than KNN in all the metrices. Still we are only getting 80% accuracy, this may be due to data imbalance. Let us look at the churn feature. As you can see below 74% of the churn is No/0’. So, the prediction in machine learning algorithms are biased more towards No than Yes.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Picture10.jpg)

#### 2.1.	Over Sampling:

To solve this problem, we will create synthetic data using over sampling to get the ‘Yes’ and ‘No’ to the same ratio. Let us look at the evaluation metrics after performing over sampling.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Customer%20Churn%20Prediction/Figures/Result2.PNG)

So, after performing over sampling, the accuracy increased a lot. XG boost is still giving better accuracy than KNN (this is expected as XG-boost is an ensemble technique). For further improving the accuracy we can use Deep Neural Network, but it takes a lot of time.

#### Important Conclusions:

•	People who have month to month contracts have most of the churn rate, but the median total charges for people who did not renew their month to month contract is smaller than the median total charges for people who did renew their month to month contract. So, we need to focus more on converting people who have month to month contracts to yearly contracts by offering better deals.

•	Even if monthly charges are higher, people with a high tenure (have stayed with telecom for a while) will most likely still renew their contract.

•	Having one type of security like Online Security has almost no effect on whether someone will have another type of security like Device Protection.

•	After performing Over sampling and hyper parameter tuning XG boost is performing way better than KNN and it is giving an accuracy of 88.2% with test data.





 


