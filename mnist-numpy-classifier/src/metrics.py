import matplotlib.pyplot as plt


def plot_losses(losses, epoches, title='Training Loss', xlabel='Epochs', ylabel='Loss'):
    plt.figure(figsize=(10, 5))
    plt.plot(epoches, losses, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()