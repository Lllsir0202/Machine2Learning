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


### Example: AND Gate

```python
Input:  [0, 0], [0, 1], [1, 0], [1, 1]
Output: [1],    [0],    [0],    [1]
```

### Some other trials
I also try to use **tanh** activation function in file ``xorgate_tanh.py``.