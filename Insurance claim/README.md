#     1.	Pilot-Study Proposal

## Problem Statement:
The task of this project is to explore the feasibility of using Machine Learning to forecast whether a customer will file a claim on their travel and to predict the value of the claim
## Predictive Task:
In solving any type of problem, selecting a specific type of predictive task plays a vital role. Predictive tasks such as classification, regression, clustering, rules mining have their own significance. For example, classification techniques are used for predicting a discrete class label output. Similarly, Regression techniques are used for predicting a continuous output and finally Clustering and rules mining are for unsupervised and market-based analysis, respectively. As problem statement refers to predicting categorical output with two categories, ‘Binary Classification’ predictive task suits better. 
## Possible Informative Features:
Informative Features are the variables required for any given problem that can sufficiently help us build an accurate predictive model. A few examples of possible features are listed below:
- Name of the travel insurance products 
- Duration of travel 
- Destination of travel 
- Mode of transport
- Number of previous insurance policies
- Number of previous insurance claims
- Previous insurance destinations and durations
- Sales of travel insurance policies 
- Commission received for travel insurance agency 
- Gender of insured 
- Age of insured 
- Marital Status of insured
- Children (Yes or No)
- Target: Claim Status 
- Target: Claim Value
## Learning Procedures:
We can get solution of the given problem by using different supervised classifier models. Here we use ‘Decision tree Classifier, Random Forest Classifier, K-Nearest Neighbors Classifier, Support Vector Machine classifier and XG-boost Classifier’. Each model contains several hyperparameters, tuning them to best parameters contribute to higher accuracy of the model. Finally, the evaluation metrics will be compared for each individual model and the model which possess high accuracy will be selected. 
Among the ML models we used for this data set, I think SVM’s and XG-boost works way better because:
- SVM’s have the powerful "kernel trick", which allows them to map the data to a very high dimension space in which the data can be separable by a hyperplane. 
- XG-boost is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It uses a gradient descent algorithm to minimize the loss. 
## Performance Evaluation:
The next step is to find out how effective is the model based on performance evaluation metrics such as ‘Confusion Matrix, Accuracy, Precision, Recall, F1 score & Area Under Curve’ for Classification problem and ‘RMSE, MSE, R-square’ for regression problem. Choice of metrics influences how the performance of machine learning algorithms is measured and compared. We can use classification performance derived from simple confusion matrix.
## Accuracy:
It is the percentage of correctly classifies instances out of all instances. It is more useful on a binary classification than multi-class classification problems because it can be less clear exactly how the accuracy breaks down across those classes.
## Confusion matrix:
The Confusion matrix contains metrics used for finding the correctness and accuracy of the model. It is a table with 4 different combinations of predicted and actual values.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/Confusion%20Matrix.PNG)

   It is extremely useful for measuring Accuracy, Recall, Precision, F1 score.
   <br />Accuracy = (TP + TN) / (TP + TN + FP + FN)
   <br />It is the ratio of the number of correct predictions to the total number of input samples
   <br />Recall/Sensitivity = TP / (TP + FN) 
   <br />Out of all the positive classes, how much we predicted accurately is referred as recall
   <br />Precision= TP / (TP + FP)
   <br />Out of all the classes, how much we predicted correctly is referred as precision.
   <br />F1-score: 2 / ((1 / Precision) + (1 / Recall))
<br />It is difficult to compare two models with low precision and high recall or vice versa. So, to make them comparable, we use F1-Score. It helps to measure Recall and Precision at the same time.
## AUROC:
AUROC is Area Under the Receiver Operating Characteristics. It is one of the most important evaluation metrics for checking any classification model’s performance. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0’s as 0’s and 1’s as 1’s. 
## MSE and RMSE:
MSE is the average squared difference between the estimated values and the actual value. It tells you how close a regression line or curve is to a set of points. RMSE is just the square root of the mean square error. 
# Report on the Investigation
## Data:
The Train dataset consists of historical data made of 1500 examples with 15 features starting from "F1" to "F15" and one output label "Class" representing whether the insured filed a claim or not. On the other hand, test dataset comprises of feature from "F1" to "F15" with missing output label class to be replaced with the output predictions. 
Any Machine Learning Project has a few steps that needs to be done for getting the output like Data Pre-processing, Feature Engineering, Model Building, Hyper-parameter Tuning and finally Evaluation of Model.
## Pre-processing & Feature Engineering:
1.	It is observed that the data contains missing values in feature F15. If you see below graph you can clearly see that there is a ‘high correlation’ between F15 and Class. So, we cannot remove this feature. 
  ![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/Correlation.PNG)
2.	Also, there are no outliers in feature F15. So, we are replacing missing values with ‘Mean’. (FYI: If there are outliers, we replace the missing values with Median).
  
  ![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/Outlier.PNG)
  
3.	As the and features are not scaled, we are ‘Normalizing’ them.
## Model Building & Hyper-parameter Tuning:
As the problem is classification, we are using ‘Decision tree Classifier, Random Forest Classifier, K-Nearest Neighbors Classifier, Support Vector Machine classifier and XG-boost Classifier’. In any Machine Learning model there are various number of hyper parameters which need to be tuned according to the given data, so that the model can fit. So, we do hyper-parameter tuning using Cross Validation. We are using ‘5-fold Grid Search Cross Validation’ which construct 5 versions of model with all the possible combinations of hyper-parameters. Taking consideration of best parameters, we have built learning models and calculated various evaluation metrics. 
## Decision tree classifier:
The understanding level of Decision Trees algorithm is so easy compared with other classification algorithms. The decision tree algorithm tries to solve the problem by using tree representation. Each internal node of the tree corresponds to an attribute, and each leaf node corresponds to a class label. The hyper-parameters used for tuning are:
- ‘Max_depth’ - Length of the longest path from the tree root to a leaf, 
- ‘Min_samples_split’ - Minimum number of samples required to split an internal node and
- ‘Min_samples_leaf’ - Minimum number of samples required to split a leaf node.
## KNN classifier:
This is also one of the simplest algorithms in ML. It classifies data based on distance and number of nearest neighbors. So, it is the only parameter used for tuning. As you can see below when K is 5, we have the highest accuracy. So, we are taking ‘n_neighbors’ as 5.
 ![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/KNN.PNG)
## Random Forest classifier:
Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction. Random Forest has nearly the same hyperparameters as a decision tree.    
- ‘Max_depth’ - Length of the longest path from the tree root to a leaf, 
- ‘Max_features’ - Number of features to consider when looking for the best split, 
- ‘Bootstrap’ - Bootstrap sample is a random sample of observations, drawn with replacement, 
- ‘Criterion’ - Gini, Entropy and
- ‘Min_samples_split’ - Minimum number of samples required to split an internal node are the parameters used for tuning. 
Support Vector Machine classifier (SVM):
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. The hyper parameters used for tuning 
- 'C' - Soft margin cost function, 
- 'Gamma' - Free parameter of the Gaussian radial basis function,
- ‘Kernel' - Rbf.
<br />I am additionally using ‘XG boost’ just to check if it gives better accuracy.
## Model Evaluation:
We can make some quick comparisons between the different approaches used to improve performance showing the returns on each. 
The following table shows the results from all the improvements we made: 
 ![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/Table1.PNG)
Comparatively SVM is high in accuracy, F1_score, Recall, Precision and AUROC. Henceforth, this learning model can be used as our final model.

## Prediction using best model:
Further step is to predict the output label "Class" in test data by using SVM learning model which gives whether the insured filed a claim or not.

## Data for predicting the value of the claim:
The Train dataset consists of historical data made of 1500 examples with 16 features starting from "F1" to "F16" and one output label "Target" representing the value of the claim. On the other hand, test dataset comprises of feature from "F1" to "F16" with missing output label class to be replaced with the output predictions.
## Pre-processing & Feature Engineering:
1.	It is observed that the data contains categorical Features ‘F4’ and ‘F15’. So, I am converting them to numerical values using ‘one-hot encoding’ because many ML algorithms cannot work with categorical data directly.
2.	As the and features are not scaled, we are ‘Normalizing’ them.
## Model Building & Hyper-parameter Tuning:
As the problem is Regression, apart from ‘Linear Regression, KNN, Random Forest and SVM’, I am also using ‘Deep Neural Network’ to get better model (for low Mean Squared Error). For hyperparameter tuning we are using ‘Randomized Search Cross Validation’ instead of Grid Search Cross Validation here because it selects random combinations to train the model and we can set search iterations based on our time and resources.
## Deep Neural Network:
I used 3 dense layers of 256 neurons each. 
- ‘Normal weight initializer’ is used to prevent layer activation outputs from exploding or vanishing during a forward pass through a deep neural network. 
- ‘RELU’ activation function is used as training a deep network with RELU tends to converge much more quickly and reliably than training with other activation functions like sigmoid. 
- ‘Adam Optimizer’ is used as it changes the learning rate and momentum depending on the loss function. So, we do not need to extensively give them, and it is too fast and converges rapidly. 
- Finally, ‘Early stopping’ is used as too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model. In the below graph you can clearly see the drop in train and test RMSE.
  ![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/RMSE.PNG)
## Model Evaluation:
We can make some quick comparisons between the different approaches used to improve performance showing the returns on each. The following table shows the results from all the improvements we made.
 ![alt text](https://github.com/Yash4850/DataScience/blob/main/Insurance claim/Figures/Table2.PNG)

Comparatively, ‘Deep Neural Network’ has low MSE and RMSE. Henceforth, this learning model can be used as our final model.

## Prediction using best model:
Further step is to predict the output label ‘Target’ in test data by using Deep Neural Network learning model which gives the value of the claim.

## References: 
1.	www.kaggle.com
2.	https://scikit-learn.org/stable/
3.	https://github.com/
4.	https://machinelearningmastery.com/
5.	https://towardsdatascience.com/









