import numpy as np
import matplotlib.pyplot as plt
from model import load_model
from utils import load_and_preprocess_mnist, predict_single_image

def evaluate():
    """
    Evaluate the trained model and display sample predictions.
    """
    # Load the trained model
    model_path = '../models/handwriting_recognition_model.h5'
    model = load_model(model_path)
    
    # Load test data
    _, (x_test, y_test) = load_and_preprocess_mnist()
    
    # Make predictions on test data
    predictions = model.predict(x_test)
    
    # Display sample predictions
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        # Randomly select a test image
        idx = np.random.randint(0, len(x_test))
        
        # Display the image
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        # Get the predicted and true labels
        predicted = np.argmax(predictions[idx])
        true_label = np.argmax(y_test[idx])
        
        # Set the title
        title_color = 'green' if predicted == true_label else 'red'
        ax.set_title(f'Pred: {predicted}\nTrue: {true_label}',
                    color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate()
