# Neural Network Game
## The objective of the game:

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rocket%20Landing%20Neural%20Netwotk/Figures/Rocket.PNG)

- Steer and apply thrust to the lander.
- Avoid hitting the outside edge or the ground.
- Safely put the lander on the target to proceed.
## Data Collection
Each run of the game is added to the same file once the game has been closed.

The data in output to the ce889_dataCollection.csv file

You need to collect data multiple time to ensure there is enough data for the NN to learn from. 

## Data Pre-processing
- Read the data using pandas
- Check for null values
- Drop Duplicates
- Give column names
- Normalize the data 
- Split the data into train(80%) and test (20%)
- Convert the data into array/matrix

## Neural Network
- Create Sigmoid function and it’s derivative(taking lambda value as 0.01)
- Initialize weights and biases to random values.
- Calculate Feed forward (dot product of inputs and hidden weights),(dot product of activation outputs and output weights)
#### Backward Propagation :
- Calculate Error (2*e)
- Calculate Local Gradient(delta)
- Calculate change/derivative of weights
- Updating the weights and biases

## Hyper Parameters
- Calculate Loss/ Cost (RMSE) (Got loss reduced to 0.026).
- Given 8 neurons in hidden layer (tried 5 to 10 neurons).
- Given 500 epochs (tried 100 to 10,000).
- Given alpha and lambda as 0.1 (tried 0.1 to 0.01).
- Given Learning rate as 0.01(tried 01 to 0.01).

![alt text](https://github.com/Yash4850/DataScience/blob/main/Rocket%20Landing%20Neural%20Netwotk/Figures/Picture1.png)

- Finally, got weights and biases.
- Gave the weights and biases in Neural Net.py, calculated forward propagation and ran the game.

## Code
```ruby

#Importing the required libraries
import pandas as pd   # for reading the csv files 
import numpy as np    # for matrix multiplications during forward and backward propagations
import matplotlib.pyplot as plt # for plotting the error plot
from sklearn.model_selection import train_test_split # for splitting the data into train and test

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
    
#Defining Sigmoid Function
class Sig:
    def sigmoid(x):  
        return 1/(1+np.exp(-x*0.01))
    def sigmoid_derivative(x):  
        return x*(1-x)
      
#Forward Propagation
class FP:
    def fwd_propagation(x_fwd_input, model):    
        Weight_hidden, bias_hidden, Weight_output, bias_output = model['w1'], model['b1'], model['w2'], model['b2']
        z1 = np.dot(Weight_hidden, x_fwd_input) +bias_hidden
        a1 = Sig.sigmoid(z1)     #activation layer
        z2 = np.dot(Weight_output, a1) + bias_output
        a2 = Sig.sigmoid(z2)
        return(a2)

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
   
#Backward Propagation
class BP:
    
    def back_propagation(model,x_train_input, a2, error, r_lambda, l_rate, epsilon):
        Weight_hidden, bias_hidden, Weight_output, bias_output, dWeight_hidden_old, dWeight_output_old = model['w1'], model['b1'], model['w2'], model['b2'], model['dw1_old'], model['dw2_old']
    
        delta2 = r_lambda *(error * Sig.sigmoid_derivative(a2))
        a1 = Sig.sigmoid(np.dot(Weight_hidden, x_train_input) +bias_hidden)  
        delta1 = r_lambda * np.dot(Weight_output.T, delta2)*Sig.sigmoid_derivative(a1)
      
        dWeight_output = np.dot(delta2, a1.T)
        dWeight_hidden = np.dot(delta1, x_train_input.T)
        dbias_output = np.sum(delta2, axis=1, keepdims=True)
        dbias_hidden = np.sum(delta1, axis=1, keepdims=True)
        
        # update the weights with the derivative (slope) of the loss function
        Weight_hidden += l_rate*dWeight_hidden + epsilon * dWeight_hidden_old
        Weight_output += l_rate*dWeight_output + epsilon * dWeight_output_old
        bias_hidden += l_rate*dbias_hidden
        bias_output += l_rate*dbias_output
        # Assign new parameters to the model
        model = { 'w1': Weight_hidden, 'b1': bias_hidden, 'w2': Weight_output, 'b2': bias_output, "dw1_old":dWeight_hidden, "dw2_old": dWeight_output}
        return model
  
 
class Build_Model:
    
    #Building Model
    def build_model(x_train_input, y_train_output,n_hidden, r_lambda, l_rate, epsilon, epochs):
    
        #initialize parameters to random values
        Weight_hidden = np.random.rand(n_hidden,2) 
        bias_hidden = np.zeros((n_hidden,1))
        Weight_output = np.random.rand(2,n_hidden) 
        bias_output = np.zeros((2,1))
    
        #gradient momentum initilizations
        dWeight_hidden_old = np.zeros((n_hidden,2))
        dWeight_output_old = np.zeros((2,n_hidden))
    
        #Declaring dictonary for storing parameters for later use
        model = {}
    
        # Assign new parameters to the model
        model = { 'w1': Weight_hidden,'b1': bias_hidden, 'w2': Weight_output, 'b2': bias_output, "dw1_old":dWeight_hidden_old, "dw2_old": dWeight_output_old}
    
        training_loss= []

        for i in range(0, epochs):
        
            #forward propagation
            a2 = FP.fwd_propagation(x_train_input, model)
            error = 2*(y_train_output - a2)
        
            #backward propagation
            model = BP.back_propagation(model,x_train_input, a2, error, r_lambda, l_rate, epsilon)
                
            rmserror  = Loss.calculate_loss(model, x_train_input, y_train_output)

            training_loss.append(rmserror)
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if i % 100 == 0:
                print("Loss after iteration %i: training loss = %f  " %(i,rmserror))
     
        return model, training_loss

# Giving the hyperparameters, training the model and finding the loss
# Here epsilon is momentum, l_rate is learning rate, r_lambda is the lambda in activation function (sigmoid)
# Tried r_lambda, epsilon and l_rate from 0.01 to 0.1, found the below values as best
model, training_loss = Build_Model.build_model(Preprocessing.x_train_input, Preprocessing.y_train_output,n_hidden= 8, epochs = 501, epsilon=0.1, r_lambda = 0.1, l_rate=0.01)
def plot_erros(training_loss):
    plt.plot(training_loss)
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('loss vs number of epochs')
    plt.legend(['training'], loc='upper right')
    plt.xlim(0, 500)
    plt.show()
plot_erros(training_loss)```
