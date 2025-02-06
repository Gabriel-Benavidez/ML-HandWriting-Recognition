# Handwritten Character Recognition System

This project implements a deep learning model for recognizing handwritten digits and characters using TensorFlow and Keras.

## Features
- Recognition of handwritten digits (0-9)
- Recognition of handwritten characters (A-Z)
- Convolutional Neural Network (CNN) architecture
- Training and evaluation scripts
- Model saving and loading capabilities

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Project Structure
```
ML-HandWriting-Recognition/
├── requirements.txt
├── src/
│   ├── train.py        # Training script
│   ├── model.py        # Model architecture
│   ├── utils.py        # Utility functions
│   └── evaluate.py     # Evaluation script
├── data/               # Dataset directory
└── models/             # Saved models directory
```

## Setup
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```
   python src/train.py
   ```

3. Evaluate the model:
   ```
   python src/evaluate.py
   ```

## License
MIT License