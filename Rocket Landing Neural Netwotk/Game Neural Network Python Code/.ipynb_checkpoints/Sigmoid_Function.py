#Defining Sigmoid Function
class Sig:
    def sigmoid(x):
        return 1/(1+np.exp(-x*0.01))
    def sigmoid_derivative(x):
        return x*(1-x)