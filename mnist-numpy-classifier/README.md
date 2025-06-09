# handwritten-digit-classifier-numpy

## Dateset
In this project, we use ``MNIST`` to **train** and **test** our model.

## Structure
This project will be origanised as following:

    mnist-numpy-classifier/
    │
    ├── data/               # Stores the MNIST dataset (downloaded or extracted here)
    │
    ├── src/                # Core implementation code
    │   ├── model.py        # MLP model implementation
    │   ├── utils.py        # Common helpers
    │   ├── train.py        # Main training script
    │   ├── activations.py  # Activation functions
    │   ├── metrics.py      # Visualization functions
    │   ├── losses.py       # Losses functions
    │   ├── optimizer.py    # Optimizer functions
    │   └── dataloader.py   # Load dataset
    │
    │
    ├── main.py             # Start train and evaluate results
    ├── README.md           # Project documentation
    │
    └── requirements.txt    # Required Python packages
