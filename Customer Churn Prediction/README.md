
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
Before going to analysis, we need to look at the data for null values and categorical variables. So, I performed a lot of pre-processing steps, but I am not mentioning them here as this report is more on explaining the above-mentioned business objectives. If you want to know more on data pre-processing steps, please refer to the Jupyter notebook (code) that I submitted as I have explained everything more clearly there.

#### 1.1.	Initial Observations:

•	Only 16% of the population are Senior Citizens and only 30% of the population have dependents.

•	10% of the population still does not have phone service. But our aim here is to retain the existing customers, not to get new customers. So, we do not focus more on this.


