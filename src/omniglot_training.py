import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import logging
import os
from datetime import datetime
from omniglot_loader import OmniglotLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OmniglotTrainer:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.input_shape = (28, 28, 1)
        self.num_classes = 1623
        
        # Create data loader
        self.data_loader = OmniglotLoader(batch_size=batch_size)
        
        # Setup model checkpoint directory
        self.checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models',
            'omniglot'
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup TensorBoard directory
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'logs',
            'omniglot',
            f'run_{datetime.now().strftime("%Y%m%d-%H%M%S")}_b{batch_size}_deeper'
        )
        os.makedirs(self.log_dir, exist_ok=True)
    
    def build_model(self) -> Sequential:
        """Build and compile the CNN model."""
        initializer = tf.keras.initializers.HeNormal()
        regularizer = tf.keras.regularizers.l2(1e-4)
        
        model = Sequential([
            Input(shape=self.input_shape),
            
            # First block
            Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Second block
            Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Third block
            Conv2D(256, (3, 3), activation='relu', padding='same',
                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same',
                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Global pooling and dense layers
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu', 
                  kernel_initializer=initializer,
                  kernel_regularizer=regularizer),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu',
                  kernel_initializer=initializer,
                  kernel_regularizer=regularizer),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax',
                  kernel_initializer=initializer)
        ])
        
        # Use a lower learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )
        
        # Compile model with label smoothing
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.1,
                from_logits=False
            ),
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ]
        )
        
        return model
    
    def train(self, epochs: int = 100):
        """Train the model on Omniglot dataset."""
        try:
            logger.info("Loading dataset...")
            train_ds, val_ds = self.data_loader.load_data()
            
            # Print dataset info
            for images, labels in train_ds.take(1):
                logger.info(f"Training batch shape: {images.shape}")
                logger.info(f"Labels shape: {labels.shape}")
                logger.info(f"Labels sum: {tf.reduce_sum(labels)}")
                logger.info(f"Labels min/max: {tf.reduce_min(labels)}/{tf.reduce_max(labels)}")
            
            logger.info("Building model...")
            model = self.build_model()
            model.summary()
            
            # Callbacks
            callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(
                        self.checkpoint_dir,
                        'model-{epoch:02d}-{val_accuracy:.4f}.keras'
                    ),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                TensorBoard(
                    log_dir=self.log_dir,
                    histogram_freq=1,
                    update_freq='epoch'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            logger.info("Starting training...")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

if __name__ == '__main__':
    # Train model
    trainer = OmniglotTrainer()
    trainer.train()
