# CIFAR-10 Image Classification (CNN Project)

This repository contains a Convolutional Neural Network (CNN) model built using TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 categories.

## Features
- Loads and preprocesses CIFAR-10 dataset
- Custom train/validation/test split
- CNN with 3 convolutional blocks
- Model training with accuracy/loss tracking
- Evaluation using test accuracy and classification report
- Includes experiments with:
  - More training epochs
  - Smaller dense layer size
  - Deeper CNN architecture

## Project Structure
```
machine.py      # Main script for model creation, training, evaluation
Report.pdf      # Full project report with experiment results and analysis
README.md       # Project documentation
```

## Model Architecture (Baseline)
- Conv2D(32) → MaxPooling2D  
- Conv2D(64) → MaxPooling2D  
- Conv2D(128) → MaxPooling2D  
- Flatten → Dense(128) → Dropout(0.5) → Dense(10 Softmax)

## Requirements
Install the necessary dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## How to Run
Run the training and evaluation script:

```bash
python3 machine.py
```

This script will:
- Train the CNN model
- Display accuracy/loss curves
- Evaluate the model on the test dataset
- Print the classification report

## Results (Summary)
- Baseline test accuracy: ~75%
- Additional experiments explored:
  - 25 training epochs
  - Dense layer reduced to 64 units
  - Added additional 256-filter convolutional block

(See **Report.pdf** for detailed graphs, metrics, and analysis.)

## License
Created for **CS 6840/4840 – Intro to Machine Learning**.  
For academic and educational use.
