import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import load_model
import logging
import tensorflow as tf
from custom_dataset import CustomDatasetGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_custom_data():
    """
    Load and preprocess the custom synthetic dataset of letters and digits.
    
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test), num_classes)
    """
    logger.info("Generating custom dataset of letters and digits...")
    
    # Initialize dataset generator
    generator = CustomDatasetGenerator()
    
    # Generate dataset with 100 samples per class
    return generator.generate_dataset(samples_per_class=100)

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_misclassified_examples(x_test, y_true, y_pred, class_names, num_examples=10):
    """
    Plot examples of misclassified images.
    """
    misclassified = np.where(y_true != y_pred)[0]
    num_examples = min(num_examples, len(misclassified))
    
    plt.figure(figsize=(20, 4))
    for i in range(num_examples):
        idx = misclassified[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model():
    """
    Evaluate the trained model on our custom dataset.
    """
    try:
        # Load custom data
        (x_train, y_train), (x_test, y_test), num_classes = load_custom_data()
        
        # Get class names
        generator = CustomDatasetGenerator()
        class_names = generator.get_class_names()
        
        # Load the trained model
        logger.info("Loading trained model...")
        try:
            model = load_model('../models/handwriting_recognition_alphanumeric.h5')
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return
        
        # Evaluate model
        logger.info("Evaluating model on custom test set...")
        metrics = model.evaluate(x_test, y_test, verbose=1)
        metrics_names = model.metrics_names
        
        print("\nTest Metrics:")
        for name, value in zip(metrics_names, metrics):
            print(f"{name}: {value:.4f}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(x_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_names))
        
        # Plot confusion matrix
        logger.info("Plotting confusion matrix...")
        plot_confusion_matrix(true_classes, predicted_classes, class_names)
        
        # Plot misclassified examples
        logger.info("Plotting misclassified examples...")
        plot_misclassified_examples(x_test, true_classes, predicted_classes, class_names)
        
        # Suggestions for improvement
        print("\nSuggestions for Improvement:")
        print("1. Increase the variety of synthetic samples")
        print("2. Add data augmentation (rotation, slight skewing)")
        print("3. Fine-tune model architecture for alphanumeric recognition")
        print("4. Collect real handwritten samples for testing")
        print("5. Use transfer learning from pre-trained models")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")

if __name__ == '__main__':
    evaluate_model()
