#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    e = np.eye(6)
    F = np.zeros((6,6))

    @qml.template
    def tcircuit(p):
        variational_circuit(p)

    @qml.qnode(dev)
    def circuit(params1,params2):
        variational_circuit(params1)
        qml.inv(tcircuit(params2))
        return qml.probs(wires=[0,1,2])
        #return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    assert np.isclose(circuit(params+np.pi/2*(e[0]+e[1]),params)[0],circuit(params,params+np.pi/2*(e[0]+e[1]))[0])

    #met_fn = qml.metric_tensor(qnode)
    #print(met_fn(params))

    for i in range(6):
        for j in range(6):
            F[i,j] = -circuit(params+np.pi/2*(e[i]+e[j]),params)[0]+circuit(params+np.pi/2*(e[i]-e[j]),params)[0]+circuit(params+np.pi/2*(-e[i]+e[j]),params)[0]-circuit(params-np.pi/2*(e[i]+e[j]),params)[0]
            # ? return < shifted | qnode >
    #print(circuit(params+np.pi/2*(e[i]+e[j]),params))

    #print((1/8.)*F)

    s = np.pi/2
    gradient = np.zeros(6)
    for i in range(6):
        gradient[i] = ( qnode(params + s*e[i]) - qnode(params - s*e[i])) / (2*np.sin(s))
    #print(gradient)
    
    #quit() 

    natural_grad = np.linalg.inv((1/8.)*F)@gradient

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
