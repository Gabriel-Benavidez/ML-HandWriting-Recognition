import os
from model import create_model, save_model
from utils import load_and_preprocess_mnist, plot_training_history
import numpy as np

def train():
    """
    Train the handwritten digit recognition model.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs('../models')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    
    # Train the model
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = model.evaluate(x_test, y_test, verbose=1)
    metrics_names = model.metrics_names
    
    print("\nTest Metrics:")
    for name, value in zip(metrics_names, metrics):
        print(f"{name}: {value:.4f}")
    
    # Make predictions on test data
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(true_classes, predicted_classes)
    
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes))
    
    # Save the model
    model_path = '../models/handwriting_recognition_model.h5'
    save_model(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot training history
    plot_training_history(history)

if __name__ == '__main__':
    train()
