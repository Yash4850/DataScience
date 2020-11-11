import numpy as np
from Sigmod_Function import Sig
class FP:
    def fwd_propagation(x_fwd_input, model):
        Weight_hidden, bias_hidden, Weight_output, bias_output = model['w1'], model['b1'], model['w2'], model['b2']
        z1 = np.dot(Weight_hidden, x_fwd_input) +bias_hidden
        a1 = Sig.sigmoid(z1)     #activation layer
        z2 = np.dot(Weight_output, a1) + bias_output
        a2 = Sig.sigmoid(z2)
        return(a2)