import numpy as np

class LogicalLearning:
    def __init__(self):
        # w1 -> weights for first layer
        self.w1 = np.random.randn(2,2)
        # b1 -> biases for first layer
        self.b1 = np.zeros((2,))
        # w2 -> weights for second layer -> means it conclude the output into a num which indicate the output is 1 or 0
        self.w2 = np.random.randn(2,1)
        # w2 -> biases for second layer
        self.b2 = np.zeros((1,))

        # The total procedure is : input-> x' = f(w1 @ input + b1) -> y = f(w2 @ x' + b2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, y):
        # Forward pass -> 
        x1 = x @ self.w1 + self.b1
        print("before sigmod x1:", x1.shape)
        a1 = self.sigmoid(x1)
        print("after sigmod x1:", a1.shape)

        x2 = a1 @ self.w2 + self.b2
        print("before sigmod x2:", x2.shape)
        a2 = self.sigmoid(x2)
        print("after sigmod x2:", a2.shape)

        loss = np.mean((y - a2.squeeze()) ** 2)
        print(loss)
        return x1, a1, x2, a2, loss

    def train(self, epochs = 5, batch_size = 4, learning_rate = 0.01):
        # This function is used to train the model
        # for epoch in range(epochs):

        return

if __name__ == "__main__":
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 0, 0, 1]

    np_x = np.array(x)
    np_y = np.array(y)
    print("Input:", np_x)
    print("Output:", np_y)
    ll = LogicalLearning()

    ll.forward(np_x, np_y)