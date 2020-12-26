import pandas as pd
import numpy as np    #for matrix multiplications during forward and backward propagations
from math import sqrt

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

        # Weights & Biases
        self.Weight_hidden = ([[-86.63571743, 64.04752951],
                                    [91.86121687, 151.09125343],
                                    [639.83781873, -278.86122779],
                                    [30.2144904, -876.00376451],
                                    [-143.53563794, -25.49789591]])
        self.Weight_output = ([[-55.12403334, -78.25352326, 66.92258235, 282.75946846,
                                     34.65361806],
                                    [-87.33920054, -37.05847893, 163.01745278, -72.6198597,
                                     -99.72260426]])
        self.bias_hidden = ([[3.57370199],
                                  [14.32379274],
                                  [-50.81281684],
                                  [96.61311179],
                                  [29.98867889]])
        self.bias_output = ([[-6.9389034],
                                  [-9.32586186]])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x*0.01))

    def fwd_propagation(self,x_fwd_input):
        z1 = np.dot(self.Weight_hidden, x_fwd_input) + self.bias_hidden
        a1 = self.sigmoid(z1)  # activation layer
        z2 = np.dot(self.Weight_output, a1) + self.bias_output
        a2 = self.sigmoid(z2)
        return (a2)

    def predict(self, input_row):
        input_list = [float(x) for x in input_row.split(",")]
        df = pd.DataFrame(np.array(input_list).reshape(-1,2), columns=["X distance to target",  "Y distance to target"])
        # Predicted Target values
        y_prediction = self.fwd_propagation(df.T)
        output = [val for list in y_prediction for val in list]
        return (output)