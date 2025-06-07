import numpy as np

DEBUG = True

class MLP:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.activation_functions = 'relu'  # 默认激活函数
        for input_dim , output_dim in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.randn(input_dim, output_dim))
            self.biases.append(np.zeros(output_dim))
            if DEBUG:
                print("input shape is ", input_dim, "output shape is ", output_dim)
    
    def activate(self, input, kind='relu'):
        if kind == 'relu':
            return np.maximum(0, input)
        elif kind == 'sigmoid':
            return 1/ (1 + np.exp(-input))
        elif kind == 'tanh':
            return np.tanh(input)
        else:
            raise ValueError(f"Unknown activation function: {kind}")

    def activate_dirivative(self, input, kind):
        if kind == 'relu':
            return np.where(input > 0, 1, 0)
        elif kind == 'sigmoid':
            return input * (1 - input)
        elif kind == 'tanh':
            return 1 - input ** 2
        else:
            raise ValueError(f"Unknown activation function: {kind}")

    def forward(self, x):
        activations = []
        zs = []
        a = x
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ w + b
            zs.append(z)
            if DEBUG:
                print("z shape is ", z.shape)
            a = self.activate(z, self.activation_functions)
            if DEBUG:
                print("a shape is ", a.shape)
            activations.append(a)

        # output layer -> use sigmoid activation
        z = a @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        if DEBUG:
            print("output z shape is ", z.shape)
        a = self.activate(z, 'sigmoid')
        if DEBUG:
            print("output a shape is ", a.shape)
        activations.append(a)

        return zs, activations

if __name__ == "__main__":
    # Example usage
    mlp = MLP([2, 4, 3, 1])
    print(len(mlp.weights))  # 应该是 3 层参数
    print(mlp.weights[0].shape)  # 应该是 (2, 4)
    print(mlp.biases[0].shape)   # 应该是 (4,)

    mlp.forward(np.array([[1, 2], [3, 4]]))  # 测试前向传播