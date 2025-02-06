import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandwritingCanvas:
    def __init__(self, model_path='../models/handwriting_recognition_alphanumeric.h5'):
        self.root = tk.Tk()
        self.root.title("Handwriting Recognition Test Interface")
        
        # Load the model
        logger.info("Loading model...")
        self.model = load_model(model_path)
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create canvas
        self.canvas_size = 280  # 10x larger than model input
        self.canvas = tk.Canvas(self.main_frame, 
                              width=self.canvas_size, 
                              height=self.canvas_size, 
                              bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_position)
        self.last_x = None
        self.last_y = None
        
        # Create buttons
        self.clear_button = tk.Button(self.main_frame, 
                                    text="Clear", 
                                    command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, pady=5)
        
        self.predict_button = tk.Button(self.main_frame, 
                                      text="Predict", 
                                      command=self.predict)
        self.predict_button.grid(row=1, column=1, pady=5)
        
        # Create prediction label
        self.prediction_label = tk.Label(self.main_frame, 
                                       text="Draw a character and click 'Predict'",
                                       font=('Arial', 14))
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Store characters for prediction
        self.characters = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    def paint(self, event):
        """Handle mouse drag events for drawing."""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, 
                                  event.x, event.y, 
                                  width=20, fill='black', 
                                  capstyle=tk.ROUND, 
                                  smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y
    
    def reset_position(self, event):
        """Reset last known position when mouse is released."""
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete('all')
        self.prediction_label.config(text="Draw a character and click 'Predict'")
    
    def preprocess_image(self):
        """Convert canvas to model input format."""
        # Create PIL image from canvas
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_size
        y1 = y + self.canvas_size
        
        # Get canvas content as image
        image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw all canvas objects onto the image
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            draw.line(coords, fill='black', width=20)
        
        # Resize to model input size
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict(self):
        """Predict the drawn character."""
        try:
            # Preprocess the image
            img_array = self.preprocess_image()
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx] * 100
            
            # Get predicted character
            predicted_char = self.characters[predicted_idx]
            
            # Update label with prediction and confidence
            self.prediction_label.config(
                text=f"Predicted: {predicted_char} (Confidence: {confidence:.2f}%)"
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            self.prediction_label.config(text="Error during prediction")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == '__main__':
    app = HandwritingCanvas()
    app.run()
