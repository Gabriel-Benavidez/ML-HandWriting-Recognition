import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OmniglotLoader:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.num_classes = 1623  # Omniglot has 1623 different characters
        self.input_shape = (28, 28, 1)
        
        # Initialize augmentation layers with more conservative rotation
        self.rotation_layer = tf.keras.layers.RandomRotation(
            factor=(-0.05, 0.05),  # Reduced to ±18° (5% of 360°)
            fill_mode='constant',
            fill_value=0.0
        )
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.15,    # Keep translation as is
            width_factor=0.15,
            fill_mode='constant',
            fill_value=0.0
        )
        self.zoom_layer = tf.keras.layers.RandomZoom(
            height_factor=(-0.15, 0.15),  # Keep zoom as is
            fill_mode='constant',
            fill_value=0.0
        )
    
    def preprocess_image(self, image, label):
        """Preprocess image and label."""
        # Convert image to float32 and normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Resize image to 28x28
        image = tf.image.resize(image, [28, 28])
        
        # Ensure image is grayscale
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        
        # Invert images (Omniglot has white characters on black background)
        image = 1.0 - image
        
        # Ensure label is in the correct range
        label = tf.cast(label, tf.int32)
        tf.debugging.assert_less(label, self.num_classes)
        tf.debugging.assert_greater_equal(label, 0)
        
        return image, label
    
    def _augment(self, image, label):
        """Apply data augmentation to the image."""
        # Ensure 4D tensor for batched images
        if len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, 0)
            
        # Random brightness
        image = tf.image.random_brightness(image, 0.2)
        
        # Random contrast
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random translation (fixed padding and crop size)
        padded = tf.image.pad_to_bounding_box(image, 2, 2, 32, 32)
        image = tf.image.random_crop(padded, [tf.shape(image)[0], 28, 28, 1])
        
        # Add random Gaussian noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        
        # Random zoom (small amount)
        shape = tf.shape(image)
        size = tf.cast(tf.cast([shape[1], shape[2]], tf.float32) * tf.random.uniform([], 0.9, 1.1), tf.int32)
        image = tf.image.resize(image, size, method='bilinear')
        image = tf.image.resize_with_crop_or_pad(image, 28, 28)
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image[0] if len(tf.shape(image)) == 4 else image, label
    
    def augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply data augmentation to training images."""
        # Random rotation
        image = self.rotation_layer(image)
        
        # Apply additional augmentations
        image, label = self._augment(image, label)
        
        return image, label
    
    def prepare_dataset(self, dataset: tf.data.Dataset, is_training: bool = False) -> tf.data.Dataset:
        """Prepare dataset for training/testing."""
        
        # Set batch size based on training/testing
        batch_size = self.batch_size if is_training else self.batch_size * 2
        
        # Define preprocessing function
        def preprocess(image, label):
            # Convert image to float32 and normalize
            image = tf.cast(image, tf.float32) / 255.0
            
            # Ensure image is grayscale and has correct shape
            if len(image.shape) == 2:
                image = tf.expand_dims(image, -1)
            
            # Convert label to one-hot encoding
            label = tf.one_hot(label, depth=1623)
            
            return image, label
        
        # Apply preprocessing
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            # Shuffle before caching for training
            dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
        
        # Cache after preprocessing but before batching
        dataset = dataset.cache()
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        
        if is_training:
            # Apply augmentation only to training data
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess Omniglot dataset."""
        try:
            logger.info("Loading Omniglot dataset...")
            
            # Load dataset using TensorFlow Datasets
            train_ds, test_ds = tfds.load(
                'omniglot',
                split=['train', 'test'],
                as_supervised=True,  # Load as (image, label) pairs
                shuffle_files=True
            )
            
            # Get number of examples
            train_size = tf.data.experimental.cardinality(train_ds).numpy()
            test_size = tf.data.experimental.cardinality(test_ds).numpy()
            logger.info(f"Training samples: {train_size}")
            logger.info(f"Test samples: {test_size}")
            
            # Check label distribution
            logger.info("Checking label distribution...")
            label_counts = {}
            for _, label in train_ds.take(1000):  # Check first 1000 samples
                label_val = label.numpy()
                label_counts[label_val] = label_counts.get(label_val, 0) + 1
            
            logger.info(f"Number of unique labels: {len(label_counts)}")
            logger.info(f"Label range: {min(label_counts.keys())} to {max(label_counts.keys())}")
            logger.info(f"Average samples per class: {sum(label_counts.values()) / len(label_counts):.2f}")
            
            # Verify a few samples
            logger.info("Verifying dataset samples...")
            for image, label in train_ds.take(1):
                logger.info(f"Sample image shape: {image.shape}")
                logger.info(f"Sample label: {label.numpy()}")
                logger.info(f"Label range: min={tf.reduce_min(label)}, max={tf.reduce_max(label)}")
            
            # Prepare datasets
            train_ds = self.prepare_dataset(train_ds, is_training=True)
            test_ds = self.prepare_dataset(test_ds, is_training=False)
            
            # Verify processed datasets
            logger.info("Verifying processed datasets...")
            for images, labels in train_ds.take(1):
                logger.info(f"Batch image shape: {images.shape}")
                logger.info(f"Batch label shape: {labels.shape}")
                logger.info(f"Labels min/max values: {tf.reduce_min(labels):.2f}/{tf.reduce_max(labels):.2f}")
                logger.info(f"Number of classes in batch: {tf.reduce_sum(tf.cast(tf.reduce_max(labels, axis=1) > 0, tf.int32))}")
            
            logger.info("Dataset loaded and preprocessed successfully!")
            return train_ds, test_ds
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

if __name__ == '__main__':
    # Test the loader
    loader = OmniglotLoader()
    try:
        train_ds, test_ds = loader.load_data()
        
        # Print dataset info
        print("\nSuccessfully loaded Omniglot dataset:")
        print(f"Number of classes: {loader.num_classes}")
        print(f"Image shape: {loader.input_shape}")
        
        # Test a batch
        for images, labels in train_ds.take(1):
            print(f"\nBatch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
