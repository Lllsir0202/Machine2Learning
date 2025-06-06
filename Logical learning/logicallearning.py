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

        loss = np.mean((y - a2) ** 2)
        print(loss)
        return x1, a1, x2, a2, loss

    def train(self, x, y, epochs = 5, batch_size = 4, learning_rate = 0.01):
        # This function is used to train the model
        for epoch in range(epochs):
            print(f"Training Epoch {epoch+1} / {epochs}")
            # output training epoch

            # First backward -> start from output layer
            # calculate the gradient of loss with respect to output layer
            x1,a1,x2,a2,loss = self.forward(x, y)
            # Now we review Senior Math
            # loss = 1/n * (y - a2) ^ 2
            # dL/da2 = - 2/n (y - a2) = 2/n (a2 - y)
            # a2 = sigmoid(x2)
            # da2/dx2 = a2 * (1 - a2)
            # dL/dx2 = dL/da2 * da2/dx2 = 2/n (a2 - y) * a2 * (1 - a2)

            # What we need is dL/dw2 and dL/db2
            # L = 1/n * (y - a2) ^ 2 = 1/n * (y - sigmoid(x2)) ^ 2 = 1/n * (y - sigmod(a1 @ w2 + b2)) ^ 2
            # so dL/dw2 = dL/dx2 * dx2/dw2
            # x2 = a1 @ w2 + b2 -> dx2/dw2 = a1
            # dL/dw2 = dL/dx2 * dx2/dw2 = dL/dx2 * a1 = 2/n (a2 - y) * a2 * (1 - a2) * a1
            # dL/db2 = dL/dx2 * dx2/db2 = dL/dx2 * 1 = 2/n (a2 - y) * a2 * (1 - a2)
            dz2 = 2 / len(y) * (a2 - y) * a2 * (1 - a2)
            print(dz2.shape)
            dw2 = a1.T @ dz2
            print(dw2.shape)
            db2 = np.sum(dz2, axis=0)
            print(db2.shape)
        return

if __name__ == "__main__":
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 0, 0, 1]

    np_x = np.array(x)
    np_y = np.array(y).reshape(-1,1)
    print("Input:", np_x)
    print("Inpit shape:", np_x.shape)
    print("Output:", np_y)
    print("Output shape:", np_y.shape)
    ll = LogicalLearning()

    ll.train(np_x, np_y)