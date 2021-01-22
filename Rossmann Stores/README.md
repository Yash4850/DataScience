# Rossmann Store Sales
## Data Description:
We are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were 
temporarily closed for refurbishment.

## Files
train.csv - historical data including Sales,
test.csv - historical data excluding Sales, sample_submission.csv - a sample submission file in the correct format, store.csv - supplemental information about the stores.

Problem Statement:
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. We need to predict sales based on their unique circumstances.
## Pre-Processing
- Merged both ‘Store’ and ‘Train’ data and ‘Store’ and ‘Test’ data
- Mapped the State Holiday ({0: 0, "0": 0, "a": 1, "b": 1, "c": 1}) for both train and test data.
- Added a few other columns like ‘Month’, ’Average Customers’ and ‘Sales per customer’ for a better understanding of data.
- Replaced missing values by calculating the median of the column.
- Replaced the null values in test data ‘Open’ column with 1 as they are all in weekdays and not in holidays and converting it to float/int.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture1.png)

- As you can see below there’s a very low correlation between ‘Promo2SinceWeek’, ‘Competition Open Since Year’, ‘Competition Open Since Month’, ’Competition Distance’. So, we dropped all these columns.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture2.png)

- As you can see, even though the store types B, C & D are less in count, their sales are very high. So, they are relevant for predicting the sales.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture3.png)

- Similarly, even though the Assortment type B are less in count, their sales are very high.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture4.png)

- Removed rows where stores are closed, as Sales are zero
- Copied the Id’s of rows in test data where store is closed and delete them, as the sales are 0. Finally, add them at the end with zero sales.

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture5.png)

- Did one-hot encoding for “Assortment, Day of Week and Store type” to convert the categorical variables to numerical variables.
- Used Standard Scaler for scaling the data.

## Deep Neural Network
- Used Weight Initialization, activation function, optimizer and 3 dense layers of 32 neurons each.
- Used Normal weight initializer to prevent layer activation outputs from exploding or vanishing during a forward pass through a deep neural network.
- Used RELU Activation function as training a deep network with RELU tends to converge much more quickly and reliably than training with sigmoid activation.
- Used Adam Optimizer as it changes the learning rate and momentum depending on the loss function. So, we don’t need to extensively give them, and it is too fast and converges rapidly.
- Added DROPOUT layer to regularize the model/network.
- Used Early stopping as too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model. 

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture6.png)

- Got the final score of 0.14

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rossmann%20Stores/Figures/Picture7.png)

