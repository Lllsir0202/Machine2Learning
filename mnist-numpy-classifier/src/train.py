from model import Model
from SGD import SGD
from losses import cross_entropy
import numpy as np
from dataloader import *
from metrics import plot_losses

def get_batches(data, labels, batch_size):
    """Yield batches of data and labels."""
    n_samples = data.shape[0]
    for i in range(0, n_samples, batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]

def train(epochs=10, batch_size=32, learning_rate=0.01, plot=False):
    # Load MNIST dataset
    train_images, train_labels, test_images, test_labels = load_dataset(flatten=True, one_hot=True)

    # Initialize model
    model = Model(input_size=784, output_size=10, activation='relu')
    # Initialize optimizer
    optimizer = SGD(learning_rate=learning_rate)
    # Losses to paint
    losses = []
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in get_batches(train_images, train_labels, batch_size):
            prob = model.forward(x_batch)
            loss = cross_entropy(prob, y_batch,mean=True)
            total_loss += loss

            model.backward(x_batch, y_batch, learning_rate)
            optimizer.step(model.parameters(), model.gradients())
        avg_loss = total_loss / (len(train_images) // batch_size)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
        if plot:
            losses.append(avg_loss)
    if plot:
        epochnum = range(1, epochs + 1)
        plot_losses(losses, epochnum)
    evaluate(model, test_images, test_labels)

def evaluate(model, test_images, test_labels):
    predict_prob = model.forward(test_images)
    predict_label = np.argmax(predict_prob, axis=1)
    true_label = np.argmax(test_labels, axis=1)
    accuracy = np.mean(predict_label == true_label)
    print(f'Accuracy: {accuracy:.4f}')