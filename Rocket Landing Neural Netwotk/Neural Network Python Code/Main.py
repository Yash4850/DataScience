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
# Here alpha is momentum, l_rate is learning rate, r_lambda is the lambda in activation function (sigmoid)
# Tried r_lambda, alpha and l_rate from 0.01 to 0.1, found the below values as best
model, training_loss = Build_Model.build_model(Preprocessing.x_train_input, Preprocessing.y_train_output,n_hidden= 8, epochs = 501, alpha=0.1, r_lambda = 0.1, l_rate=0.01)

# hidden weights
print("self.Weight_hidden =", repr(model["w1"]), "\nself.Weight_output = ",repr(model["w2"]), "\nself.bias_hidden = ",repr(model["b1"]),"\nself.bias_output = ",repr(model["b2"]))

#Plotting the loss vs Epochs
def plot_erros(training_loss):
    plt.plot(training_loss)
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('loss vs number of epochs')
    plt.legend(['training'], loc='upper right')
    plt.xlim(0, 500)
    plt.show()
plot_erros(training_loss)