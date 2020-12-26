import pandas as pd
import numpy as np    #for matrix multiplications during forward and backward propagations
import matplotlib.pyplot as plt  #for plotting the error plot
from sklearn.model_selection import train_test_split #for splitting the data into train and test
import math  #for divisions
from math import sqrt
from Sigmod_Function import Sig
#calculating loss
class Loss:
    def calculate_loss(model, x_trainloss, y_trainloss):
       #calling model prediction
        Weight_hidden, bias_hidden, Weight_output, bias_output = model['w1'], model['b1'], model['w2'], model['b2']
        z1 = np.dot(Weight_hidden, x_trainloss) +bias_hidden
        a1 = Sig.sigmoid(z1)  #activation layer
        z2 = np.dot(Weight_output, a1) + bias_output
        a2 = Sig.sigmoid(z2)
        rmserror = np.mean(np.square(y_trainloss - a2))
        z2 = np.dot(Weight_output, a1) + bias_output
        a2 = Sig.sigmoid(z2)
        return rmserror