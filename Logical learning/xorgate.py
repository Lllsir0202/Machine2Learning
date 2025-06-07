import numpy as np
from LogicalLearning import LogicalLearning

DEBUG = False

class XORGATE(LogicalLearning):
    def __init__(self):
        self.w1 = np.random.randn(2,2)
        self.b1 = np.zeros((2,))
        self.w2 = np.random.randn(2,1)
        self.b2 = np.zeros((1,))

    def forward(self, x, y):
        x1 = x @ self.w1 + self.b1
        a1 = self.sigmoid(x1)
        x2 = a1 @ self.w2 + self.b2
        a2 = self.sigmoid(x2)
        loss = np.mean((y - a2) ** 2)

        if DEBUG:
            print("x1 shape:", x1.shape)
            print("a1 shape:", a1.shape)
            print("x2 shape:", x2.shape)
            print("a2 shape:", a2.shape)
            print("loss:", loss)

        return x1, a1, x2, a2, loss
    
    def train(self, x, y, epochs = 10000, learning_rate = 0.1):
        losses = []
        for epoch in range(epochs):
            if (epoch + 1) % 1000 == 0:
                print(f"Training Epoch {epoch+1} / {epochs}")
            # Output layer
            x1, a1, x2, a2, loss = self.forward(x, y)
            dz2 = 2 / len(y) * (a2 - y) * self.sigmoid_derivative(a2)
            dw2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0)
            if DEBUG:
                print("dz2 shape:", dz2.shape)
                print("dw2 shape:", dw2.shape)
                print("db2 shape:", db2.shape)

            # Hidden layer
            dz1 = dz2 @ self.w2.T * self.sigmoid_derivative(a1)
            dw1 = x.T @ dz1
            db1 = np.sum(dz1, axis=0)
            if DEBUG:
                print("dz1 shape:", dz1.shape)
                print("dw1 shape:", dw1.shape)
                print("db1 shape:", db1.shape)

            # Update weights and biases
            self.w2 = self.w2 - learning_rate * dw2
            self.b2 = self.b2 - learning_rate * db2
            self.w1 = self.w1 - learning_rate * dw1
            self.b1 = self.b1 - learning_rate * db1
            losses.append(loss)

        self.plot_draw(np.arange(1, epochs + 1), losses)
        print("Finial Loss:", losses[-1])
        return
    
    def predict(self, x):
        x1 = x @ self.w1 + self.b1
        a1 = self.sigmoid(x1)
        x2 = a1 @ self.w2 + self.b2
        a2 = self.sigmoid(x2)
        return (a2 > 0.5).astype(int)

if __name__ == "__main__":
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 1, 1, 0]

    np_x = np.array(x)
    np_y = np.array(y).reshape(-1, 1)

    xorgate = XORGATE()
    # xorgate.forward(np_x, np_y)

    xorgate.train(np_x, np_y)
    predictions = xorgate.predict(np_x)
    print("Predictions:\n", predictions)