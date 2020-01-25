from miniflow import *
import numpy as np

def test_add():
    x, y = Input(), Input()
    f = Add(x, y)
    g = Add(f, x)

    feed_dict = {x: 10, y: 5}
    sorted_nodes = topological_sort(feed_dict)

    output = forward_pass(sorted_nodes)
    print("({} + {}) + {} = {} (according to miniflow)\n".format(feed_dict[x],
                                                               feed_dict[y],
                                                               feed_dict[x],
                                                               output))

def test_multiply():
    x, y, z = Input(), Input(), Input()
    f = Mul(x, y, z)

    feed_dict = {x: 10, y: 5, z: 6}
    sorted_nodes = topological_sort(feed_dict)

    output = forward_pass(sorted_nodes)
    print("{} x {} x {} = {} (according to miniflow)\n".format(feed_dict[x],
                                                             feed_dict[y],
                                                             feed_dict[z],
                                                             output))

def test_linear_layer():
    inputs, weights, bias = Input(), Input(), Input()
    f = Linear(inputs, weights, bias)

    feed_dict = {
        inputs: np.array([6, 14, 3]),
        weights: np.array([0.5, 0.25, 1.4]),
        bias: np.array([2])
    }
    graph = topological_sort(feed_dict)

    output = forward_pass(graph)
    print("Output of Linear node:", output, "\t...should be 12.7\n")

    X, W, b = Input(), Input(), Input()
    f = Linear(X, W, b)
    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}
    graph = topological_sort(feed_dict)

    output = forward_pass(graph)
    print("Output of linear node:\n", output)
    print("\nOutput should be:\n [[-9., 4.],\n [-9., 4.]]\n")

def test_sigmoid():
    X, W, b = Input(), Input(), Input()
    f = Linear(X, W, b)
    g = Sigmoid(f)
    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}
    graph = topological_sort(feed_dict)

    output = forward_pass(graph)
    print("Output of sigmoid layer:\n", output)
    print("\nOutput should be:\n [[1.23394576e-04   9.82013790e-01]\n [1.23394576e-04   9.82013790e-01]]\n")

def test_mse():
    y, a = Input(), Input()
    cost = MSE(y, a)
    y_ = np.array([1, 2, 3])
    a_ = np.array([4.5, 5, 10])

    feed_dict = {y: y_, a: a_}
    graph = topological_sort(feed_dict)

    forward_pass(graph)
    print("Output of MSE node:\n", cost.value)
    print("\nExpected output:\n 23.4166666667\n")

def test_backward():
    X, W, b = Input(), Input(), Input()
    y = Input()
    f = Linear(X, W, b)
    a = Sigmoid(f)
    cost = MSE(y, a)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2.], [3.]])
    b_ = np.array([-3.])
    y_ = np.array([1, 2])

    feed_dict = {
        X: X_,
        y: y_,
        W: W_,
        b: b_,
    }
    graph = topological_sort(feed_dict)
    forward_pass(graph)
    backpropagation(graph)


    # return the gradients for each Input
    gradients = [t.gradients[t] for t in [X, y, W, b]]
    print("Gradients: \n", gradients, "\n")
    print("Expected output:",
          "\n[array([[ -3.34017280e-05,  -5.01025919e-05],",
          "\n\t[ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],",
          "\n\t[ 1.9999833]]), array([[  5.01028709e-05],",
          "\n\t[  1.00205742e-04]]), array([ -5.01028709e-05])]\n")

if __name__ == "__main__":
    test_add()
    test_multiply()
    test_linear_layer()
    test_sigmoid()
    test_mse()
    test_backward()
