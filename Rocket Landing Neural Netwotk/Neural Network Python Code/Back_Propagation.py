import numpy as np
from Data_Preprocessing import Preprocessing
from Sigmod_Function import Sig
from Forward_Propagation import FP
# Backward Propagation
class BP:

    def back_propagation(model, x_train_input, a2, error, r_lambda, l_rate, epsilon):
        Weight_hidden, bias_hidden, Weight_output, bias_output, dWeight_hidden_old, dWeight_output_old = model['w1'], \
                                                                                                         model['b1'], \
                                                                                                         model['w2'], \
                                                                                                         model['b2'], \
                                                                                                         model[
                                                                                                             'dw1_old'], \
                                                                                                         model[
                                                                                                             'dw2_old']

        delta2 = r_lambda * (error * Sig.sigmoid_derivative(a2))
        a1 = Sig.sigmoid(np.dot(Weight_hidden, x_train_input) + bias_hidden)
        delta1 = r_lambda * np.dot(Weight_output.T, delta2) * Sig.sigmoid_derivative(a1)

        dWeight_output = np.dot(delta2, a1.T)
        dWeight_hidden = np.dot(delta1, x_train_input.T)
        dbias_output = np.sum(delta2, axis=1, keepdims=True)
        dbias_hidden = np.sum(delta1, axis=1, keepdims=True)

        # update the weights with the derivative (slope) of the loss function
        Weight_hidden += l_rate * dWeight_hidden + epsilon * dWeight_hidden_old
        Weight_output += l_rate * dWeight_output + epsilon * dWeight_output_old
        bias_hidden += l_rate * dbias_hidden
        bias_output += l_rate * dbias_output
        # Assign new parameters to the model
        model = {'w1': Weight_hidden, 'b1': bias_hidden, 'w2': Weight_output, 'b2': bias_output,
                 "dw1_old": dWeight_hidden, "dw2_old": dWeight_output}
        return model