import numpy as np
from LogicalLearning import LogicalLearning

class ANDGATE(LogicalLearning):
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

    def forward(self, x, y):
        # Forward pass -> 
        x1 = x @ self.w1 + self.b1
        # print("before sigmod x1:", x1.shape)
        a1 = self.sigmoid(x1)
        # print("after sigmod x1:", a1.shape)

        x2 = a1 @ self.w2 + self.b2
        # print("before sigmod x2:", x2.shape)
        a2 = self.sigmoid(x2)
        # print("after sigmod x2:", a2.shape)

        loss = np.mean((y - a2) ** 2)
        # print(loss)
        return x1, a1, x2, a2, loss

    def train(self, x, y, epochs = 10000, batch_size = 4, learning_rate = 0.1):
        # This function is used to train the model
        losses = []
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
            dz2 = 2 / len(y) * (a2 - y) * self.sigmoid_derivative(a2)
            # print(dz2.shape)
            dw2 = a1.T @ dz2
            # print(dw2.shape)
            db2 = np.sum(dz2, axis=0)
            # print(db2.shape)

            # Now we need to calculate the gradient of loss with respect to first layer
            # dL/dx1 = dL/dx2 * dx2/dx1 -> x2 = a1 @ w2 + b2 = sigmod(x1) @ w2 + b2
            # dx2/dx1 = dx2/da1 * da1/dx1 = w2.T * a1 * (1 - a1)
            # dz1 = dz2 @ self.w2.T * a1 * (1 - a1)
            dz1 = dz2 @ self.w2.T * self.sigmoid_derivative(a1)
            # print(dz1.shape)
            # dw1 = dL/dw1 = dL/dx1 * dx1/dw1 = dz1 * dx1/dw1
            # x1 = w1 @ x + b1 -> dx1/dw1 = x.T
            # dw1 = x.T @ dz1
            dw1 = x.T @ dz1
            # print(dw1.shape)
            # db1 = dL/db1 = dL/dx1 * dx1/db1 = dz1 * 1 = dz1
            db1 = np.sum(dz1, axis=0)
            # print(db1.shape)

            # Next We need to update the weights and biases
            self.w2 = self.w2 - learning_rate * dw2
            self.b2 = self.b2 - learning_rate * db2
            self.w1 = self.w1 - learning_rate * dw1
            self.b1 = self.b1 - learning_rate * db1
            # print("Weights and biases updated")
            losses.append(loss)

        x_axis = np.arange(1, epochs + 1)
        self.plot_draw(x_axis, losses)
        return

    def predict(self, x):
        # First layer
        a1 = self.sigmoid(x @ self.w1 + self.b1)
        a2 = self.sigmoid(a1 @ self.w2 + self.b2)
        return (a2 > 0.5).astype(int)

if __name__ == "__main__":
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 0, 0, 1]

    np_x = np.array(x)
    np_y = np.array(y).reshape(-1,1)
    print("Input:", np_x)
    print("Inpit shape:", np_x.shape)
    print("Output:", np_y)
    print("Output shape:", np_y.shape)
    ll = ANDGATE()

    ll.train(np_x, np_y)

    # ll.predict(np_x)
    print("Predictions:\n", ll.predict(np_x))