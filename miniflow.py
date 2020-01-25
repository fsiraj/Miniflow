import numpy as np

class Node(object):
    def __init__(self, in_nodes=[]):
        # lists of input and output nodes
        self.in_nodes = in_nodes
        self.out_nodes = []
        # appends this node to out_nodes of all input nodes
        for n in self.in_nodes:
            n.out_nodes.append(self)
        self.gradients = {}
        self.value = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self):
        # list of input nodes empty
        Node.__init__(self)
        # value set during topological_sort
    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.out_nodes:
            running_grad = n.gradients[self]
            self.gradients[self] += running_grad

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 0
        for n in self.in_nodes:
            self.value += n.value

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for n in self.in_nodes:
            self.value *= n.value

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        inputs = self.in_nodes[0].value
        weights = self.in_nodes[1].value
        bias = self.in_nodes[2].value
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            # partial with respect to linear node
            running_grad = n.gradients[self]
            # partial with respect to this node's inputs
            self.gradients[self.in_nodes[0]] += np.dot(running_grad, self.in_nodes[1].value.T)
            # partial with respect to this node's weights
            self.gradients[self.in_nodes[1]] += np.dot(self.in_nodes[0].value.T, running_grad)
            # partial with respect to this node's bias
            self.gradients[self.in_nodes[2]] += np.sum(running_grad, axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self, input):
        Node.__init__(self, [input])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        x = self.in_nodes[0].value
        self.value = self._sigmoid(x)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            running_grad = n.gradients[self]
            sig_x = self.value
            self.gradients[self.in_nodes[0]] = sig_x * (1 - sig_x) * running_grad


class MSE(Node):
    def __init__(self, y, y_hat):
        Node.__init__(self, [y, y_hat])

    def forward(self):
        y = self.in_nodes[0].value.reshape(-1, 1)
        y_hat = self.in_nodes[1].value.reshape(-1, 1)
        self.value = np.mean((y-y_hat)**2)

    def backward(self):
        m = self.in_nodes[0].value.shape[0]
        y = self.in_nodes[0].value.reshape(-1, 1)
        y_hat = self.in_nodes[1].value.reshape(-1, 1)
        self.gradients[self.in_nodes[0]] = (2 / m) * (y - y_hat)
        self.gradients[self.in_nodes[1]] = (-2 / m) * (y - y_hat)


def forward_pass(graph):
    for n in graph:
        n.forward()
    output = graph[-1].value
    return output

def backpropagation(graph):
    for n in graph[::-1]:
        n.backward()

def sgd_step(parameters, lr=0.001):
    for p in parameters:
        partial = p.gradients[p]
        p.value -= lr * partial


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.out_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.out_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
