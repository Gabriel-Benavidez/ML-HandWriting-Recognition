import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def create_model(input_shape=(28, 28, 1), num_classes=1623):
    """
    Create a CNN model for handwritten character recognition.
    
    Architecture:
    - Input layer: 28x28x1 (grayscale images)
    - 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool + Dropout)
    - Flatten layer
    - Dense layers (128, 256 neurons)
    - Output layer with softmax activation
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of classes to predict (1623 for Omniglot)
        
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        
        # Dense hidden layer
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(
            num_classes,
            activation='softmax'
        )
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1,
            from_logits=False  # We're using softmax activation
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
    return model

def model_summary(model):
    """
    Print model summary and architecture details.
    
    Args:
        model: Keras model
    """
    print("\nModel Architecture:")
    print("==================")
    model.summary()
    
    print("\nLayer Details:")
    print("=============")
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}: {layer.__class__.__name__}")
        print(f"Config: {layer.get_config()}")

def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained Keras model
        filepath (str): Path to save the model
    """
    model.save(filepath)
    print(f"\nModel saved to: {filepath}")

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        model: Loaded Keras model
    """
    return models.load_model(filepath)

if __name__ == '__main__':
    # Create and display model architecture
    model = create_model()
    model_summary(model)
