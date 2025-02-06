import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_mnist():
    """
    Load MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test), num_classes
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = 10
    
    return (x_train, y_train), (x_test, y_test), num_classes

def preprocess_data(x_train, y_train, x_test, y_test, num_classes):
    """
    Preprocess the dataset:
    1. Normalize pixel values
    2. Reshape images
    3. Convert labels to one-hot encoding
    
    Args:
        x_train, y_train, x_test, y_test: Raw data
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        tuple: Preprocessed (x_train, y_train), (x_test, y_test)
    """
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return (x_train, y_train), (x_test, y_test)

def visualize_samples(x_data, y_data, num_samples=5, one_hot=True):
    """
    Visualize sample images and their labels.
    
    Args:
        x_data: Image data
        y_data: Labels (one-hot encoded or not)
        num_samples: Number of samples to display
        one_hot: Whether labels are one-hot encoded
    """
    # Convert one-hot encoded labels back to single numbers if necessary
    if one_hot:
        y_data = np.argmax(y_data, axis=1)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    
    # Randomly select samples
    indices = np.random.randint(0, len(x_data), num_samples)
    
    for i, ax in enumerate(axes):
        # Get the image and label
        img = x_data[indices[i]]
        label = y_data[indices[i]]
        
        # Remove the channel dimension if present
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img.squeeze()
        
        # Display the image
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test), num_classes = load_mnist()
    
    print("\nOriginal data shapes:")
    print(f"Training data: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    
    print("\nVisualizing original samples:")
    visualize_samples(x_train, y_train, num_samples=5, one_hot=False)
    
    print("\nPreprocessing data...")
    (x_train, y_train), (x_test, y_test) = preprocess_data(
        x_train, y_train, x_test, y_test, num_classes
    )
    
    print("\nPreprocessed data shapes:")
    print(f"Training data: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    
    print("\nVisualizing preprocessed samples:")
    visualize_samples(x_train, y_train, num_samples=5, one_hot=True)
    
    # Print value ranges
    print("\nData value ranges:")
    print(f"Training data min: {x_train.min():.3f}, max: {x_train.max():.3f}")
    print(f"Test data min: {x_test.min():.3f}, max: {x_test.max():.3f}")
