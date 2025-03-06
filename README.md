# Image Classification with CNN on CIFAR-10 Dataset

This repository contains the code and report for an image classification task using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. The goal of this project was to build, train, and improve a CNN model to classify images into one of 10 categories (e.g., airplanes, cars, birds, etc.). The project involved experimenting with various modifications to the baseline model to improve its performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Modifications and Improvements](#modifications-and-improvements)
- [Results](#results)
- [Challenges](#challenges)
- [How to Run the Code](#how-to-run-the-code)
- [Conclusion](#conclusion)

---

## Project Overview
The objective of this project was to build and improve a CNN model for image classification on the CIFAR-10 dataset. The project involved the following steps:
1. **Loading and Preprocessing the Data**: The CIFAR-10 dataset was loaded, normalized, and one-hot encoded.
2. **Building a Baseline CNN Model**: A simple CNN model with 2 convolutional layers, max pooling, and fully connected layers was created.
3. **Training and Evaluating the Model**: The baseline model was trained for 10 epochs, and its performance was evaluated on the test set.
4. **Improving the Model**: Several modifications were applied to the baseline model to improve its performance, including adding dropout layers, increasing the depth of the model, using data augmentation, and adding batch normalization layers.
---

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The classes include:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Model Architecture
### Baseline Model
- **Convolutional Layers**: 2 layers with 32 filters each.
- **Max Pooling Layers**: Applied after each convolutional layer.
- **Fully Connected Layers**: One fully connected layer with ReLU activation.
- **Output Layer**: Softmax activation for classification.
- **Optimizer**: Adam optimizer.
- **Loss Function**: Categorical cross-entropy.
- **Metrics**: Accuracy.

### Improved Models
1. **Improved Model 1**: Added dropout layers with a rate of 0.5 to reduce overfitting.
2. **Improved Model 2**: Increased the number of filters in the convolutional layers from 32 to 64 and added an additional convolutional layer.
3. **Improved Model 3**: Applied data augmentation techniques (rotation, flipping, zooming) to increase the diversity of the training data.
4. **Improved Model 4**: Added batch normalization layers after each convolutional layer to stabilize training.

---

## Results
The performance of the baseline and improved models was evaluated based on test accuracy. The results are summarized below:

| Model Version        | Modifications Applied          | Test Accuracy (%) |
|----------------------|--------------------------------|-------------------|
| Baseline Model       | None (original configuration)  | 69%               |
| Improved Model 1     | Added dropout layers           | 66%               |
| Improved Model 2     | Increased depth and filters    | 70%               |
| Improved Model 3     | Used data augmentation         | 73%               |
| Improved Model 4     | Added batch normalization      | 70%               |

---

## Challenges
Several challenges were encountered during the project:
1. **Poor Image Resolution**: The low quality of the images made it difficult for the model to learn useful features, especially for similar-looking classes.
2. **Small Dataset Size**: The dataset size was relatively small, making the model more prone to overfitting.
3. **Long Training Time**: Adding more filters and using data augmentation increased the training time, slowing down the experimentation process.

---

## How to Run the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/image-classification-cnn.git
   cd image-classification-cnn
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Conclusion
This project demonstrated the process of building and improving a CNN model for image classification on the CIFAR-10 dataset. Various modifications were applied to the baseline model, and their impact on performance was analyzed. The best-performing model achieved a test accuracy of 73% using data augmentation. The challenges encountered, such as poor image resolution and long training times, provided valuable insights into the complexities of working with image datasets in machine learning.
