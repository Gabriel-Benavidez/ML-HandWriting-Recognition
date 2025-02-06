import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import logging
import cv2
from scipy.ndimage import rotate
import random
from scipy.ndimage import map_coordinates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDatasetGenerator:
    def __init__(self, image_size=(28, 28)):
        self.image_size = image_size
        self.characters = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.num_classes = len(self.characters)
        self.fonts = [
            'Arial', 'Times New Roman', 'Courier New',
            'Georgia', 'Verdana', 'Helvetica'
        ]
        
    def apply_random_transform(self, image):
        """Apply random transformations to make the image more realistic."""
        # Convert PIL Image to numpy array
        img = np.array(image)
        
        # Random stroke width variation (0.8 to 1.2 of original)
        kernel_size = random.randint(1, 3)
        if random.random() > 0.5:
            img = cv2.dilate(img, np.ones((kernel_size, kernel_size), np.uint8))
        else:
            img = cv2.erode(img, np.ones((kernel_size, kernel_size), np.uint8))
        
        # Random rotation (-20 to 20 degrees)
        angle = random.uniform(-20, 20)
        img = rotate(img, angle, reshape=False)
        
        # Random perspective transform
        if random.random() > 0.5:
            height, width = img.shape
            src_pts = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
            dst_pts = np.float32([[0, 0], 
                                [width-1 + random.uniform(-5, 5), random.uniform(-5, 5)],
                                [random.uniform(-5, 5), height-1 + random.uniform(-5, 5)],
                                [width-1 + random.uniform(-5, 5), height-1 + random.uniform(-5, 5)]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img = cv2.warpPerspective(img, M, (width, height))
        
        # Random elastic deformation
        if random.random() > 0.7:
            dx = cv2.GaussianBlur(np.random.randn(28, 28) * 2, (7, 7), 0)
            dy = cv2.GaussianBlur(np.random.randn(28, 28) * 2, (7, 7), 0)
            x, y = np.meshgrid(np.arange(28), np.arange(28))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            img = map_coordinates(img, indices, order=1).reshape(28, 28)
        
        # Random scaling (0.7 to 1.3)
        scale = random.uniform(0.7, 1.3)
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
        
        # Add padding if necessary
        if new_size[0] > self.image_size[0] or new_size[1] > self.image_size[1]:
            img = cv2.resize(img, self.image_size)
        else:
            pad_h = (self.image_size[0] - new_size[0]) // 2
            pad_v = (self.image_size[1] - new_size[1]) // 2
            img = np.pad(img, ((pad_v, pad_v), (pad_h, pad_h)), mode='constant', constant_values=255)
            img = img[:self.image_size[0], :self.image_size[1]]
        
        # Random shear (-0.3 to 0.3)
        shear = random.uniform(-0.3, 0.3)
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, self.image_size)
        
        # Add random noise (0 to 25)
        noise = np.random.normal(0, random.uniform(0, 25), img.shape)
        img = np.clip(img + noise, 0, 255)
        
        # Random brightness variation (0.6 to 1.4)
        brightness = random.uniform(0.6, 1.4)
        img = np.clip(img * brightness, 0, 255)
        
        # Random contrast (0.6 to 1.4)
        contrast = random.uniform(0.6, 1.4)
        mean = np.mean(img)
        img = np.clip((img - mean) * contrast + mean, 0, 255)
        
        # Add random dropouts to simulate broken strokes
        if random.random() > 0.8:
            mask = np.random.random(img.shape) > 0.95
            img[mask] = 255
        
        return Image.fromarray(img.astype('uint8'))
        
    def generate_character_image(self, char, font_size=24):
        """Generate an image of a single character with variations."""
        # Create a new image with white background
        image = Image.new('L', (32, 32), color=255)
        draw = ImageDraw.Draw(image)
        
        try:
            # Try different fonts
            font_name = random.choice(self.fonts)
            try:
                font = ImageFont.truetype(font_name, font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size and position to center it
            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (32 - text_width) // 2
            y = (32 - text_height) // 2
            
            # Draw the character
            draw.text((x, y), char, fill=0, font=font)
            
            # Apply random transformations
            image = self.apply_random_transform(image)
            
            # Resize to desired dimensions
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            return np.array(image)
        
        except Exception as e:
            logger.error(f"Error generating image for character {char}: {str(e)}")
            return None
    
    def generate_dataset(self, samples_per_class=100):
        """Generate a dataset with synthetic characters."""
        logger.info(f"Generating synthetic dataset with {samples_per_class} samples per class...")
        
        x_data = []
        y_data = []
        
        for idx, char in enumerate(self.characters):
            logger.info(f"Generating samples for character: {char}")
            
            for _ in range(samples_per_class):
                img = self.generate_character_image(char)
                if img is not None:
                    x_data.append(img)
                    y_data.append(idx)
        
        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Normalize pixel values
        x_data = x_data.astype('float32') / 255.0
        
        # Reshape to include channel dimension
        x_data = x_data.reshape(-1, 28, 28, 1)
        
        # Convert labels to one-hot encoding
        y_data = tf.keras.utils.to_categorical(y_data, self.num_classes)
        
        # Split into training and testing sets
        indices = np.random.permutation(len(x_data))
        split_idx = int(len(indices) * 0.8)  # 80% training, 20% testing
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        x_train = x_data[train_indices]
        y_train = y_data[train_indices]
        x_test = x_data[test_indices]
        y_test = y_data[test_indices]
        
        logger.info(f"Dataset generated successfully:")
        logger.info(f"Training samples: {len(x_train)}")
        logger.info(f"Testing samples: {len(x_test)}")
        logger.info(f"Number of classes: {self.num_classes}")
        
        return (x_train, y_train), (x_test, y_test), self.num_classes
    
    def get_class_names(self):
        """Return the list of class names (characters)."""
        return self.characters
