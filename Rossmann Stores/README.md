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
