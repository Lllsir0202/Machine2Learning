import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

class MLP:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.activation_functions = 'relu'  # 默认激活函数
        for input_dim , output_dim in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim))  # He initialization
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

    def activate_derivative(self, z, kind):
        if kind == 'relu':
            return np.where(z > 0, 1, 0)
        elif kind == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        elif kind == 'tanh':
            t = np.tanh(z)
            return 1 - t ** 2
        else:
            raise ValueError(f"Unknown activation function: {kind}")

    def forward(self, x):
        activations = [x]
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

    def backward(self, x, y, zs, activations,  learnging_rate = 0.01):
        # It's not good to use forward in backward function
        # zs, activations = self.forward(x)
        loss = np.mean((activations[-1] - y) ** 2)
        
        grads_w = []
        grads_b = []

        # From back to front , we calculate the gradients
        # and then we need to record the gradients
        
        # Firstly , we need to calculate the gradient of the output layer
        delta = (activations[-1] - y) * self.activate_derivative(zs[-1], 'sigmoid')
        dw_output = activations[-2].T @ delta
        db_output = np.sum(delta, axis=0)
        grads_w.append(dw_output)
        grads_b.append(db_output)

        # Then we calculate

        # Because we have already calculated the output layer, we start from the second last layer
        # The activation[l] means the activation of the l-th layer,
        for l in reversed(range(len(self.weights) - 1)):
            z = zs[l]
            delta = (delta @ self.weights[l + 1].T) * self.activate_derivative(z, self.activation_functions)
            dw = activations[l].T @ delta
            db = np.sum(delta, axis=0)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learnging_rate * grads_w[i]
            self.biases[i] = self.biases[i] - learnging_rate * grads_b[i]
        
        if DEBUG:
            print("Update weights and biases")
        return loss

    def plot_draw(self, x_axis, losses):
        plt.plot(x_axis, losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()

    def train(self, x, y, kind='relu', epochs = 10000, learning_rate = 0.01):
        self.activation_functions = kind
        losses = []
        for epoch in range(epochs):
            zs, activations = self.forward(x)
            loss  = self.backward(x, y, zs, activations, learning_rate)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            losses.append(loss)
        
        self.plot_draw(np.arange(1, epochs + 1), losses)
        
    def predict(self, x):
        _, activations = self.forward(x)
        return (activations[-1] > 0.5).astype(int)

if __name__ == "__main__":
    # Example usage
    # mlp = MLP([2, 4, 3, 1])
    # print(len(mlp.weights))  # 应该是 3 层参数
    # print(mlp.weights[0].shape)  # 应该是 (2, 4)
    # print(mlp.biases[0].shape)   # 应该是 (4,)

    # mlp.forward(np.array([[1, 2], [3, 4]]))  # 测试前向传播

    DEBUG = False  # 可以关闭调试信息以更清晰观察训练

    # 训练数据（异或）
    x_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_train = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    mlp = MLP([2, 4, 3, 1])
    mlp.train(x_train, y_train, kind='tanh', epochs=100, learning_rate=0.1)

    print("Predictions:")
    print(mlp.predict(x_train))
    # print("Rounded predictions:")
    # print((mlp.predict(x_train) >= 0.5).astype(int))