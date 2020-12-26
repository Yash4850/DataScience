import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessing:
    df = pd.read_csv('ce889_dataCollection.csv',header = None)
    # for dropping the duplicates
    df.drop_duplicates(inplace = True)
    # checking for null values
    print("Checking For Null values", df.isnull().sum())
    # Giving column names
    df.columns = ["X distance to target","Y distance to target","New Vel Y","New Vel X"]
    # normalization
    data_min = df.min()
    data_max = df.max()
    normalizeddata = (df - df.min()) / (df.max() - df.min())
    # spliting data
    training , testing= train_test_split(normalizeddata, test_size = 0.2)
    print("Shape of Training Data", training.shape)
    print("Shape of Testing Data", testing.shape)
    # Splitting inputs & outputs for training, validation & testing data
    x_train_input = np.array(training[["X distance to target","Y distance to target"]]).T
    y_train_output = np.array(training[["New Vel Y","New Vel X"]]).T

    x_testing_input = np.array(testing[["X distance to target","Y distance to target"]]).T
    y_testing_output = np.array(testing[["New Vel Y","New Vel X"]]).T