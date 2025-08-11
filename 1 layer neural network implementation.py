import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
from planar_utils import (
    plot_decision_boundary,
    sigmoid,
    load_planar_dataset,
)

# loading dataset(check the planar_utils.py)
X, Y = load_planar_dataset()

# m (number of training examples)
m = X.shape[1]

# n_x(number of input features) , n_y(size of output) , n_h(size of hidden layer)
def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return(n_x , n_h , n_y)




# initializing parameters (w1 & b1 for the first layer [hidden layer] of network & w2,b2 for the last layer of NN which it's input is A1(n_h , m))
def initialize_parameters(n_x, n_h, n_y):

    scale_factor = 0.01
    W1 = np.random.randn(n_h, n_x) * scale_factor
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * scale_factor
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# forward prop
def forprop(params , X ):
    W1=params['W1']
    b1=params['b1']
    W2=params['W2']
    b2=params['b2']
    Z1 = np.dot(W1 , X) + b1
    A1 = np.tanh(Z1)
    Z2=np.dot(W2 , A1) + b2
    A2= sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return (cache , A2)

# cost computer
def cost_comp(A2 , Y , params):

    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)


    cost = float(np.squeeze(cost))  

    return cost

# backward prop 
def backprop(params , cache , X , Y):
    
    A1=cache['A1']
    A2=cache['A2']

    W1 = params['W1']
    W2 = params['W2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (
        1 - np.power(A1, 2)
    )
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


# parameter updater:
def update_parameters(params, grads, learning_rate):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters



#  neural network model:
def nn_model(X,Y,n_h, iteration , learning_rate , print_cost=False ):
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    params= initialize_parameters(n_x , n_h , n_y)

    for iter in range(iteration):
        cache , A2 = forprop(params , X)
        if iter % 100 == 0 and print_cost == True:
            cost = cost_comp(A2 , Y , params )
            print(f"cost afret iteration {iter}:{cost}" )
        grads = backprop(params , cache , X , Y)
        params = update_parameters(params , grads , learning_rate)
    return params

# predictor func
def predict(parameters, X):
    _ , A2 = forprop(parameters , X)
    predictions = A2 > 0.5

    return predictions


updated_parameters = nn_model(X , Y , 4 , 10000 , 1.2 , True)
prediction = predict(updated_parameters , X)

# testing our 1 layer model by different number of hidden unit and plotting the result (this part may take about 2 minutes to run! be patient.)

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50 ]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title("Hidden Layer of size %d" % n_h)
    parameters = nn_model(X, Y, n_h, 5000, 1.2 )
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T))
        / float(Y.size)
        * 100
    )
    plt.show()
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))





        


