#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    # Initialize the device
    dev = qml.device("default.qubit", wires=NODES)

    # Draw the graph - optional
    # from matplotlib import pyplot as plt
    # nx.draw(graph, with_labels=True)
    # plt.show()

    # fix cost and mixer hamiltonians
    #https://pennylane.readthedocs.io/en/stable/code/api/pennylane.qaoa.cost.max_independent_set.html
    cost_h, mixer_h = qml.qaoa.max_independent_set(graph, constrained=True)

    # apply time evolution by building cost and mixer layers
    # inputs are the Hamiltonians above and parameters to optimise
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)
    
    # the actual QAOA variational circuit
    # N_LAYERS repetitions of qaoa_layer
    @qml.qnode(dev)
    def circuit(params):
        qml.layer(qaoa_layer, N_LAYERS, params[0], params[1])
        return qml.probs(wires=range(NODES))

    #get the probabilities of each basis state
    probs=circuit(params)     
    #convert to bitstring and get positions of 1-bits   
    bitstring=np.argmax(probs)
    lex_bitstring=format(bitstring, "06b")
    for i,b in enumerate(lex_bitstring):
        if int(b)==1:
            max_ind_set.append(i)
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
