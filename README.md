# Crop-prediction

This project fine-tunes the VGG16 deep learning model for a custom image classification task with 5 output classes. Using transfer learning, the pre-trained VGG16 architecture is adapted for the dataset, enabling efficient training and high accuracy on smaller datasets.

Project Overview
Model Architecture: VGG16 (pre-trained on ImageNet).
Dataset: Custom dataset with 5 classes.
Training: Transfer learning is applied by freezing the pre-trained layers and adding a new fully connected layer.
Objective: Classify images into one of the 5 categories.

Features
Utilizes transfer learning to leverage the pre-trained weights of VGG16.
Adds a custom output layer with a softmax activation for multi-class classification.
Trains only the new layers, keeping the original VGG16 weights intact.
Model Architecture
The architecture of the modified VGG16 model includes:

VGG16 Backbone:
Convolutional layers with pre-trained weights from ImageNet.
Non-trainable layers to retain learned features.
Custom Fully Connected Layer:
Flatten layer to convert 2D feature maps to a 1D feature vector.
Dense layer with 5 neurons (number of classes) and softmax activation.

Requirements
Python 3.7 or later
Libraries:
TensorFlow
Keras
NumPy
Matplotlib
