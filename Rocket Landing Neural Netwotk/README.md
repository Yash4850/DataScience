# Neural Network Game
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
- Finally, got weights and biases.
- Gave the weights and biases in Neural Net.py, calculated forward propagation and ran the game.

