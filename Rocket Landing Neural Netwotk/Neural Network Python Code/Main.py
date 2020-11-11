import pandas as pd
import numpy as np    #for matrix multiplications during forward and backward propagations
import matplotlib.pyplot as plt  #for plotting the error plot
from sklearn.model_selection import train_test_split #for splitting the data into train and test
import math  #for divisions
from math import sqrt
import seaborn as sns
from Data_Preprocessing import Preprocessing
from Sigmod_Function import Sig
from Forward_Propagation import FP
from Loss import Loss
from Back_Propagation import BP
from Build_Model import Build_Model


# Giving the hyperparameters, training the model and finding the loss
model, training_loss = Build_Model.build_model(Preprocessing.x_train_input, Preprocessing.y_train_output,n_hidden= 5, epochs = 101, epsilon=0.1, r_lambda = 0.1, l_rate=0.01)

#printing weights and biases
print("Weights", model["w1"],model["w2"])
print("Biases", model["b1"],model["b2"])

#Plotting the loss vs Epochs
def plot_erros(training_loss):
    plt.plot(training_loss)
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('loss vs number of epochs')
    plt.legend(['training'], loc='upper right')
    plt.xlim(0, 1000)
    plt.show()
plot_erros(training_loss)

#Predicted Target values
y_prediction = FP.fwd_propagation(Preprocessing.x_testing_input, model)
y_prediction_df = pd.DataFrame(y_prediction.T,columns=["Predicted Vel Y","Predicted Vel X"])

#RMSerror for normalized data
rmserror_test = np.mean(np.square( (Preprocessing.y_testing_output.T) - y_prediction_df))
print("RMS Error", rmserror_test)

#denormalize predicted data
y_prediction_df['Predicted Vel Y'] = Preprocessing.data_min['New Vel Y'] + y_prediction_df['Predicted Vel Y']*(Preprocessing.data_max['New Vel Y'] - Preprocessing.data_min['New Vel Y'])
y_prediction_df['Predicted Vel X'] = Preprocessing.data_min['New Vel X'] + y_prediction_df['Predicted Vel X']*(Preprocessing.data_max['New Vel X'] - Preprocessing.data_min['New Vel X'])
print(y_prediction_df.head())