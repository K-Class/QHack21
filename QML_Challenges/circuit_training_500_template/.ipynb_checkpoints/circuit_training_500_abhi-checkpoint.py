#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

#ANS = [1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]

def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []
    
    # Fix random seed
    np.random.seed(0)
    
    # Hyperparameters
    n_wires = 3
    n_layers = 3
    batch_size = 5
    #opt = qml.NesterovMomentumOptimizer(0.01)
    opt = qml.AdamOptimizer()
    n_iter = 80
    
    # QHACK #
    
    # Initialize the device
    dev = qml.device("default.qubit", wires=n_wires)
    
    # ----------------------------------------------------------------------------
    # Source : https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    # ----------------------------------------------------------------------------
    def layer(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])
    
    @qml.qnode(dev)
    def circuit(weights, x=None):
        qml.QubitStateVector(x, wires=[0, 1])

        for W in weights:
            layer(W)

        return [qml.expval(qml.PauliZ(i)) for i in range(3)]


    def variational_classifier(var, angles):
        weights = var[0]
        bias = var[1]
        return circuit(weights, x=angles) + bias
    
    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss
    
    def one_hot(lbl):
        ret = None
        if lbl == -1:
            ret = [-1, 1, 1]
        elif lbl == 0:
            ret = [1, -1, 1]
        else:
            ret = [1, 1, -1]
        return ret
    
    
    def one_hot_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            l = one_hot(l)
            loss = loss + np.sum(np.subtract(l, p)**2)
        
        loss = loss / len(labels)
        return loss

    def cost(weights, features, labels):
        predictions = [variational_classifier(weights, f) for f in features]
        return one_hot_loss(labels, predictions)
    
    def discretize(out):
        return np.argmin(out)-1
        
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    
    # Parameter initialization
    var_init = (0.01 * np.random.randn(n_layers, n_wires, 3), 0.0)
    
    # Pad zero to the last dimension to use 2 wires
    # Source : https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    X_train_pad = np.c_[X_train, np.zeros((len(X_train), 1))]
    X_test_pad = np.c_[X_test, np.zeros((len(X_test), 1))]
    
    # Normalization
    # Source : https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    train_norm = np.sqrt(np.sum(X_train_pad ** 2, -1))
    test_norm = np.sqrt(np.sum(X_test_pad ** 2, -1))
    
    X_train_norm = (X_train_pad.T / train_norm).T
    X_test_norm = (X_test_pad.T / test_norm).T
    
    # Source : https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    var = var_init
    for it in range(n_iter):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train_norm[batch_index]
        Y_batch = Y_train[batch_index]
        
        var = opt.step(lambda v: cost(v, X_batch, Y_batch), var)
    # QHACK #
    predictions = [discretize(variational_classifier(var, x)) for x in X_test_norm]
    
    #print("Accuracy : {0}".format(np.sum(np.equal(ANS, predictions))/ len(ANS)))

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
