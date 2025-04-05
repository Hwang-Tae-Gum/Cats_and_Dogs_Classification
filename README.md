# Cats_and_Dogs_Classification

## Description
This project implements a binary image classification model using PyTorch to distinguish between cats and dogs. 
A custom Convolutional Neural Network (CNN) based on the Inception architecture is built from scratch. 
The model is trained and evaluated on a balanced dataset of cat and dog images, with the goal of achieving accurate classification performance.

## Dataset Information
- Total Images: 24,998
- Cats: 12,499
- Dogs: 12,499
Dataset Split:
- 80% of the data is used for training
- 20% is used for testing
Labeling:
- Cat → 0
- Dog → 1
Structure:
The dataset is stored in two folders: Cat/ and Dog/
Custom PyTorch Dataset class is used for loading and labeling the images.

## Model Architecture
The model uses a custom Inception module, inspired by GoogLeNet. It includes:
1. Inception Block:
- Parallel branches with 1x1, 3x3, 5x5 convolutions and a max-pooling path
- Each branch includes BatchNorm and ReLU activations
2. Overall Model Structure:
- Initial layers: Conv2d → BatchNorm → ReLU → MaxPool
- Two Inception modules
- AdaptiveAvgPool2d → Fully Connected (Linear) layer → Output logits (2 classes)

## Hyperparameters
```table
Parameter	Value
Optimizer	Adam
Learning Rate	0.001
Batch Size	16
Epochs	20
Scheduler	ReduceLROnPlateau (based on validation loss)
Input Image Size	224 x 224
Data Augmentation	Resize, RandomHorizontalFlip, Normalize
Loss Function	CrossEntropyLoss
```

## Improvements and Future Updates
- Apply transfer learning using pretrained models (e.g., ResNet, EfficientNet)
- Extend data augmentation (rotation, color jittering, etc.)
- Add EarlyStopping and K-Fold Cross Validation

## Conclusion
This project demonstrates the process of building a deep learning model for classifying cat and dog images using a custom CNN based on the Inception architecture. 
It includes data preprocessing, model design, training, and evaluation using PyTorch, making it a suitable hands-on example for solving binary image classification tasks. 
In the future, model performance can be further improved through transfer learning, advanced data augmentation, cross-validation, and early stopping techniques.
