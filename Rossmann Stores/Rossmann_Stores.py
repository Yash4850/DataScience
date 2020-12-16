"""
Submitted by Group 11:
Yaswanth - 2003287
Mohamad - 2004060
Sreelakshmi - 2004055
Abhay - 2004349
"""

import numpy as np # linear algebra
import pandas as pd # data processing
import math
import seaborn as sns
from pandasql import sqldf
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

#for removing warnings
import warnings
warnings.filterwarnings('ignore')

def data_preprocessing():
    # reading the data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    store_df = pd.read_csv("store.csv")

    # Data Preprocessing
    train['Date'] = pd.to_datetime(train['Date'])

    # Create Year and Month columns
    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))

    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))

    # Combining and Mapping a,b and c type stores so as to reduce the bias
    train["StateHoliday"] = train["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
    test["StateHoliday"] = test["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

    # Merging train and store_df
    train_store = train.merge(store_df, left_on=['Store'], right_on=['Store'], how='left')

    # Merging test and store_df
    test_store = test.merge(store_df, left_on=['Store'], right_on=['Store'], how='left')

    # Checking for null values
    train_store.isnull().sum()

    # Train_Store Week of year and calculating promo
    train_store['WeekOfYear'] = pd.DatetimeIndex(train_store['Date']).weekofyear
    train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
                               (train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0
    train_store['PromoOpen'] = train_store.PromoOpen.apply(lambda x: x if x > 0 else 0)
    train_store.loc[train_store.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Test_Store Week of year and calculating promo
    test_store['WeekOfYear'] = pd.DatetimeIndex(test_store['Date']).weekofyear
    test_store['PromoOpen'] = 12 * (test_store.Year - test_store.Promo2SinceYear) + \
                              (train_store.WeekOfYear - test_store.Promo2SinceWeek) / 4.0
    test_store['PromoOpen'] = test_store.PromoOpen.apply(lambda x: x if x > 0 else 0)
    test_store.loc[test_store.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Creating new Competition Open column for both train and test
    train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + (
                train_store.Month - train_store.CompetitionOpenSinceMonth)
    test_store['CompetitionOpen'] = 12 * (test_store.Year - train_store.CompetitionOpenSinceYear) + (
                test_store.Month - test_store.CompetitionOpenSinceMonth)

    # Replacing null values with median
    med_comp_month = train_store['PromoOpen'].astype('float').median(axis=0)
    train_store['PromoOpen'].replace(np.nan, math.floor(med_comp_month), inplace=True)

    med_comp_month = train_store['PromoOpen'].astype('float').median(axis=0)
    test_store['PromoOpen'].replace(np.nan, math.floor(med_comp_month), inplace=True)

    med_comp_month = train_store['CompetitionOpen'].astype('float').median(axis=0)
    train_store['CompetitionOpen'].replace(np.nan, math.floor(med_comp_month), inplace=True)

    med_comp_month = train_store['CompetitionOpen'].astype('float').median(axis=0)
    test_store['CompetitionOpen'].replace(np.nan, math.floor(med_comp_month), inplace=True)

    # Creating new columns Average Customers and Sales Per Customer
    avg_customer = sqldf(
        """
        SELECT
        Store,
        DayOfWeek,
        sum(case when Customers is not null then Sales/Customers else 0 end) as SpC,
        round(avg(Customers)) Avg_Customers
        from train_store
        group by Store,DayOfWeek
        """
    )

    test_store = sqldf(
        """
        SELECT
        t.*,
        ac.SpC,
        ac.Avg_Customers
        from test_store t
        left join avg_customer ac on t.Store = ac.Store and t.DayOfWeek = ac.DayOfWeek
        """
    )
    train_store = sqldf(
        """
        SELECT
        t.*,
        ac.SpC,
        ac.Avg_Customers
        from train_store t
        left join avg_customer ac on t.Store = ac.Store and t.DayOfWeek = ac.DayOfWeek
        """
    )

    # Create dummy varibales for DayOfWeek
    train_dummies = pd.get_dummies(train_store['DayOfWeek'], prefix='Day')
    train_dummies.drop(['Day_7'], axis=1, inplace=True)

    test_dummies = pd.get_dummies(test_store['DayOfWeek'], prefix='Day')
    test_dummies.drop(['Day_7'], axis=1, inplace=True)

    train_store = train_store.join(train_dummies)
    test_store = test_store.join(test_dummies)

    # Create dummy varibales for Assortment
    train_store_dummies = pd.get_dummies(train_store['Assortment'], prefix='Assortment')
    train_store_dummies.drop(['Assortment_c'], axis=1, inplace=True)

    test_store_dummies = pd.get_dummies(test_store['Assortment'], prefix='Assortment')
    test_store_dummies.drop(['Assortment_c'], axis=1, inplace=True)

    train_store = train_store.join(train_store_dummies)
    test_store = test_store.join(test_store_dummies)

    # Create dummy varibales for Storetype
    train_store_dummies = pd.get_dummies(train_store['StoreType'], prefix='StoreType')
    train_store_dummies.drop(['StoreType_d'], axis=1, inplace=True)

    test_store_dummies = pd.get_dummies(test_store['StoreType'], prefix='StoreType')
    test_store_dummies.drop(['StoreType_d'], axis=1, inplace=True)

    train_store = train_store.join(train_store_dummies)
    test_store = test_store.join(test_store_dummies)

    # Dropping unnecessary columns from train and test set
    train_store.drop(
        ['Customers', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'CompetitionDistance', 'Promo2SinceWeek',
         'Promo2SinceYear', 'PromoInterval', 'WeekOfYear', 'Year', 'StoreType', 'Assortment', 'Date'], axis=1,
        inplace=True)
    train_store['Open'] = train_store['Open'].astype(float)
    test_store.drop(
        ['Year', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'CompetitionDistance', 'Promo2SinceWeek',
         'Promo2SinceYear', 'PromoInterval', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'WeekOfYear',
         'WeekOfYear', 'StoreType', 'Assortment', 'Date'], axis=1, inplace=True)

    # Checking the null values of Open column
    print(test_store[test_store['Open'].isnull()])

    # fill NaN values in test with Open=1
    test_store["Open"].fillna(1, inplace=True)

    # Dropping DayOfWeek
    train_store.drop(['DayOfWeek'], axis=1, inplace=True)
    test_store.drop(['DayOfWeek'], axis=1, inplace=True)

    # remove all rows(store,date) that were closed
    train_store = train_store[train_store["Open"] != 0]

    # Saving id's of those stores which were closed so we can put 0 in their respective sales column
    closed_ids = test_store["Id"][test["Open"] == 0].values

    # remove all rows(store,date) that were closed
    test_store = test_store[test_store["Open"] != 0]

    # Resetting the index
    test_store = test_store.reset_index()
    test_store.drop(["index"], axis=1, inplace=True)

    train = train_store.drop(["Store", "Open"], axis=1)
    test = test_store.drop([ "Store", "Open"], axis=1)
    return(train, test,closed_ids)

def load_train_data(scaler_x, scaler_y):
    '''
    Transform train data set and separate a test dataset to validate the model in the end of training and normalize data
    '''
    X_train = train.drop(["Sales"], axis=1) # Features
    y_train = np.array(train["Sales"]).reshape((len(X_train), 1)) # Targets
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    return (X_train, y_train), (X_test, y_test)

def load_test_data():
    '''
    Remove column of predictions and normalize data of submission test data set.
    '''
    X_test = test.drop(["Id"], axis=1) # Features
    X_test = StandardScaler().fit_transform(X_test)

    return X_test
def rmspe_val(y_true, y_predicted):
    '''
    RMSPE calculus to validate evaluation metric about the model
    '''
    return np.sqrt(np.mean(np.square((y_true - y_predicted) / y_true), axis=0))[0]
def rmse(y_true, y_predicted):
    '''
    RMSE calculus to use during training phase
    '''
    return K.sqrt(K.mean(K.square(y_predicted - y_true)))
def rmspe(y_true, y_predicted):
    '''
    RMSPE calculus to use during training phase
    '''
    return K.sqrt(K.mean(K.square((y_true - y_predicted) / y_true), axis=-1))

# Model Building
# We took Relu activation function as it is from 0 to infinity
def create_model():
    '''
    Create a neural network
    '''
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.2)) # We are dropping a few neurons for generalizing the model
    model.add(Dense(32, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(32, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear", kernel_initializer='normal'))
    adam = Adam(lr=1e-3, decay=1e-3)

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=[rmse, rmspe])

    return model
# Hyperparameters and load data to train the model
train, test, closed_ids = data_preprocessing()
batch_size = 32
nb_epoch = 1

scaler_x = StandardScaler()
scaler_y = StandardScaler()

print('Loading data...')
(X_train, y_train), (X_test, y_test) = load_train_data(scaler_x, scaler_y)

print('Build model...')
model = create_model()
model.summary()

print('Fit model...')
filepath="rossmann.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
callbacks_list = [checkpoint, early_stopping]
log = model.fit(X_train, y_train, validation_split=0.20, batch_size=batch_size, epochs=nb_epoch, shuffle=True, callbacks=callbacks_list)

# Predicting the sales
test_data = load_test_data()
predict = model.predict(test_data)
predict = scaler_y.inverse_transform(predict)

pred = pd.DataFrame(predict,columns = ['Sales'])
submission = pd.concat([test['Id'],pred],axis=1)
# Creating closed stores dataframe
Closed_Stores = pd.DataFrame(closed_ids,columns = ['Id'])
print(Closed_Stores.shape)
Closed_Stores['Sales'] = 0
submission = submission.append(Closed_Stores)
# Converting it to csv
submission.to_csv('submission.csv', index=False)