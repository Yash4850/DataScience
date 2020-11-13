import numpy as np  # for matrix multiplications during forward and backward propagations
import math  # for divisions
from math import sqrt
from Data_Preprocessing import Preprocessing
from Sigmod_Function import Sig
from Forward_Propagation import FP
from Loss import Loss
from Back_Propagation import BP

class Build_Model:

    # Building Model
    def build_model(x_train_input, y_train_output, n_hidden, r_lambda, l_rate, epsilon, epochs):

        # initialize parameters to random values
        Weight_hidden = np.random.rand(n_hidden, 2)
        bias_hidden = np.zeros((n_hidden, 1))
        Weight_output = np.random.rand(2, n_hidden)
        bias_output = np.zeros((2, 1))

        # gradient momentum initilizations
        dWeight_hidden_old = np.zeros((n_hidden, 2))
        dWeight_output_old = np.zeros((2, n_hidden))

        # Declaring dictonary for storing parameters for later use
        model = {}

        # Assign new parameters to the model
        model = {'w1': Weight_hidden, 'b1': bias_hidden, 'w2': Weight_output, 'b2': bias_output,
                 "dw1_old": dWeight_hidden_old, "dw2_old": dWeight_output_old}

        training_loss = []

        for i in range(0, epochs):

            # forward propagation
            a2 = FP.fwd_propagation(x_train_input, model)
            error = 2 * (y_train_output - a2)

            # backward propagation
            model = BP.back_propagation(model, x_train_input, a2, error, r_lambda, l_rate, epsilon)

            rmserror = Loss.calculate_loss(model, x_train_input, y_train_output)

            training_loss.append(rmserror)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if i % 300 == 0:
                print("Loss after iteration %i: training loss = %f  " % (i, rmserror))

        return model, training_loss