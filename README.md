Used for learning Machine Learning

# Logical Learning
## AND Gate with Neural Network

A minimal two-layer neural network implemented using **only NumPy and core Python**, designed for learning the basics of **machine learning** and **neural networks**.

This network is trained to simulate the logic **AND** function through **simple forward and backward propagation** without any deep learning frameworks.

---

### Features

- Built **from scratch** using only `numpy`
- Implements:
  - Forward pass
  - Backpropagation
  - Gradient descent
- Uses the **sigmoid activation function**
- Learns the behavior of the **AND logic gate**
- Includes **loss plotting** using `matplotlib`


### Example: AND Gate

```python
Input:  [0, 0], [0, 1], [1, 0], [1, 1]
Output: [0],    [0],    [0],    [1]
```

## XOR Gate with Neural Network

A minimal two-layer neural network implemented using **only NumPy and core Python**, designed for learning the basics of **machine learning** and **neural networks**.

This network is trained to simulate the logic **XOR** function through **simple forward and backward propagation** without any deep learning frameworks.

---

### Features

- Built **from scratch** using only `numpy`
- Implements:
  - Forward pass
  - Backpropagation
  - Gradient descent
- Uses the **sigmoid activation function**
- Learns the behavior of the **XOR logic gate**
- Includes **loss plotting** using `matplotlib`


### Example: XOR Gate

```python
Input:  [0, 0], [0, 1], [1, 0], [1, 1]
Output: [1],    [0],    [0],    [1]
```

### Some other trials
I also try to use **tanh** activation function in file ``xorgate_tanh.py``.

# MLP(Multilayer Perceptron)
A simple, extensible multilayer perceptron (MLP) implemented purely in NumPy, designed to support arbitrary fully connected architectures such as [2, 4, 3, 1], allowing it to learn complex functions like XOR.

This MLP supports configurable hidden layers, activation functions, and basic gradient descent optimization.
##  Features
Fully customizable network architecture (e.g., [input, hidden1, hidden2, ..., output])

- Built entirely with NumPy

- Supports activation functions: ReLU, tanh, sigmoid

- Manually implemented:

    Forward propagation

    Backward propagation

    Mean squared error (MSE)

- Visualizes training loss with matplotlib

- Works well for small binary tasks like XOR with `tanh` activation function

- Built-in prediction support


### Example: XOR Gate

```python
Input:  [0, 0], [0, 1], [1, 0], [1, 1]
Output: [1],    [0],    [0],    [1]
```

I suppose it is easy to train models to solve other problems but I haven't try more.