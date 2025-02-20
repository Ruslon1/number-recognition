# MNIST Digit Classification with CNN

This project demonstrates the implementation of a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset. The project is written in Python using PyTorch.

## Project Structure

- **`cnn.py`**: Defines the CNN architecture.
- **`dataset.py`**: Prepares the MNIST dataset and creates data loaders for training and testing.
- **`training.py`**: Implements the training loop, model checkpointing, and evaluation on the test set.
- **`testing.py`**: Loads an external image, preprocesses it, and uses the trained model to predict the digit.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- Pillow (PIL)

Install dependencies with:

```bash
pip install torch torchvision pillow
```

## Dataset

The project uses the MNIST dataset:
- Automatically downloaded to the `../data` directory.
- Normalized to a range of [-1, 1].

## How to Run

### Training the Model

1. Ensure the MNIST dataset is available or downloaded by running the script.
2. Run the `training.py` file to start training:

   ```bash
   python training.py
   ```

   A checkpoint of the model is saved in `../checkpoint.pth`.

### Testing the Model

1. Place an image (`image.png`) in the parent directory (`../`).
   - The image should be grayscale and 28x28 pixels.
2. Run `testing.py` to predict the digit:

   ```bash
   python testing.py
   ```

   The predicted digit is printed to the console.

## Key Features

- **Custom CNN Architecture**:
  - Two convolutional layers.
  - ReLU activations and max pooling.
  - Fully connected layers for classification into 10 digit classes.

- **Training Features**:
  - Adam optimizer with learning rate 0.001.
  - CrossEntropyLoss for multi-class classification.
  - Checkpointing for resuming training.

- **Evaluation**:
  - Accuracy computation on the test dataset.

- **Image Prediction**:
  - External image preprocessing for compatibility with the trained model.
