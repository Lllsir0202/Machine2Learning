import numpy as np

DEBUG = True

class MLP:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for input_dim , output_dim in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.randn(input_dim, output_dim))
            self.biases.append(np.zeros(output_dim))
            if DEBUG:
                print("input shape is ", input_dim, "output shape is ", output_dim)
    
    def forward(self, x):
        self.activations = []
        a = x
        # for w,b in range(zip(self.weights, self.biases)):
            

if __name__ == "__main__":
    # Example usage
    mlp = MLP([2, 4, 3, 1])
    print(len(mlp.weights))  # 应该是 3 层参数
    print(mlp.weights[0].shape)  # 应该是 (2, 4)
    print(mlp.biases[0].shape)   # 应该是 (4,)