import numpy as np
import tensorflow as tf
import logging
import os
import gzip
from typing import Tuple, Dict
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMNISTLoader:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.num_classes = 47  # EMNIST balanced has 47 classes
        self.input_shape = (28, 28, 1)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'emnist')
        
        # File paths for EMNIST dataset
        self.files = {
            'train_images': os.path.join(self.data_dir, 'emnist-balanced-train-images-idx3-ubyte.gz'),
            'train_labels': os.path.join(self.data_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz'),
            'test_images': os.path.join(self.data_dir, 'emnist-balanced-test-images-idx3-ubyte.gz'),
            'test_labels': os.path.join(self.data_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')
        }
    
    def download_dataset(self):
        """Download EMNIST dataset using Kaggle API."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            logger.info("Downloading EMNIST dataset from Kaggle...")
            
            # Download dataset using Kaggle API
            subprocess.run([
                'kaggle', 'datasets', 'download',
                'crawford/emnist', 
                '--path', self.data_dir,
                '--unzip'
            ], check=True)
            
            # Move files to correct location
            src_dir = os.path.join(self.data_dir, 'gzip')
            for filename in os.listdir(src_dir):
                if filename.startswith('emnist-balanced-') and filename.endswith('.gz'):
                    src = os.path.join(src_dir, filename)
                    dst = os.path.join(self.data_dir, filename)
                    shutil.move(src, dst)
            
            # Clean up
            shutil.rmtree(src_dir)
            
            logger.info("Dataset downloaded and extracted successfully!")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            logger.error("Please make sure you have the Kaggle API installed and configured:")
            logger.error("1. Install Kaggle: pip install kaggle")
            logger.error("2. Get your Kaggle API key from https://www.kaggle.com/account")
            logger.error("3. Place the kaggle.json file in ~/.kaggle/")
            raise
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def load_idx_file(self, filepath: str, is_image: bool = True) -> np.ndarray:
        """Load IDX file format."""
        try:
            with gzip.open(filepath, 'rb') as f:
                # Skip magic number and dimension info
                if is_image:
                    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
                    data = data.reshape(-1, 28, 28, 1)  # Add channel dimension
                else:
                    data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
                return data
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {str(e)}")
            raise
    
    def check_files_exist(self) -> bool:
        """Check if all required dataset files exist."""
        missing_files = []
        for name, filepath in self.files.items():
            if not os.path.exists(filepath):
                missing_files.append(name)
        
        if missing_files:
            logger.info("Missing dataset files. Downloading...")
            try:
                self.download_dataset()
            except Exception as e:
                logger.error(f"Failed to download dataset: {str(e)}")
                return False
        return True
    
    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess EMNIST Balanced dataset."""
        try:
            logger.info("Loading EMNIST Balanced dataset...")
            
            # Check if all files exist
            if not self.check_files_exist():
                raise FileNotFoundError("Failed to download dataset files")
            
            # Load train data
            x_train = self.load_idx_file(self.files['train_images'])
            y_train = self.load_idx_file(self.files['train_labels'], is_image=False)
            
            # Load test data
            x_test = self.load_idx_file(self.files['test_images'])
            y_test = self.load_idx_file(self.files['test_labels'], is_image=False)
            
            logger.info(f"Training samples: {len(x_train)}")
            logger.info(f"Test samples: {len(x_test)}")
            
            # Normalize images
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Convert to TensorFlow datasets
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            
            # Apply preprocessing and batching
            train_ds = self._prepare_dataset(train_ds, is_training=True)
            test_ds = self._prepare_dataset(test_ds, is_training=False)
            
            logger.info("Dataset loaded and preprocessed successfully!")
            return train_ds, test_ds
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _prepare_dataset(self, dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
        """Prepare dataset for training/testing."""
        # Convert labels to one-hot
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, self.num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            # Shuffle and augment training data
            dataset = dataset.shuffle(10000)
            dataset = dataset.map(
                lambda x, y: (self._augment_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to training images."""
        # Random rotation (-10 to 10 degrees)
        angle = tf.random.uniform([], -10, 10) * np.pi / 180
        image = tf.keras.layers.RandomRotation(angle)(image)
        
        # Random width/height shift
        image = tf.keras.layers.RandomTranslation(0.1, 0.1)(image)
        
        # Random zoom
        image = tf.keras.layers.RandomZoom(0.1)(image)
        
        return image
    
    def get_class_mapping(self) -> Dict[int, str]:
        """Get mapping from class indices to characters."""
        # EMNIST balanced dataset mapping
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
        return {i: char for i, char in enumerate(chars)}

if __name__ == '__main__':
    # Test the loader
    loader = EMNISTLoader()
    try:
        train_ds, test_ds = loader.load_data()
        
        # Print dataset info
        print("\nSuccessfully loaded EMNIST dataset:")
        print(f"Number of classes: {loader.num_classes}")
        print(f"Image shape: {loader.input_shape}")
        
        # Print class mapping
        class_mapping = loader.get_class_mapping()
        print("\nClass mapping:")
        for idx, char in sorted(class_mapping.items()):
            print(f"{idx}: {char}", end=", ")
            if (idx + 1) % 10 == 0:
                print()
                
        # Test a batch
        for images, labels in train_ds.take(1):
            print(f"\nBatch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
