import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
# print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.train import train



if __name__ == "__main__":
    train(epochs=10, batch_size=32, learning_rate=0.01, plot=True)