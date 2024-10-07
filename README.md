# FER_2024 - Facial Expression Recognition with PyTorch

This project implements a **Facial Expression Recognition (FER)** model using **PyTorch**. It classifies images into three emotions: **Happy, Sad, and Surprise**. The model is trained using a custom dataset and fine-tuned to provide high accuracy in emotion detection.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to build a **convolutional neural network (CNN)** that can classify facial images into one of three emotions: **Happy**, **Sad**, and **Surprise**. The training pipeline supports custom datasets, and the model's performance is evaluated after each epoch.

### Key Components:
- **Model Training**: Trains the model using labeled images of facial expressions.
- **Evaluation Module**: Evaluates the model performance on test images and provides a summary of accuracy.
- **Visualization**: Provides loss and accuracy graphs after training.

## Features
- **PyTorch-based model** for FER.
- Supports multi-class classification with **CrossEntropyLoss**.
- **CosineAnnealingLR** scheduler for learning rate adjustment.
- Model checkpoints saved with a date suffix for versioning.
- Summarizes model accuracy during training and testing after each epoch.
  
## Dataset
The dataset consists of images in three categories:
- **Happy**
- **Sad**
- **Surprise**

Each category is stored in separate subfolders within the `root_dir`. The dataset is split into training and validation sets (80-20 split).

### Example folder structure:


## Model Architecture
The model is based on a **convolutional neural network (CNN)** designed to extract key features from facial images. The architecture can be configured through the `cfg` file, making it flexible for different tasks.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/VictorCardenasG/fer_2024.git
    cd fer_2024
    ```

2. Create a virtual environment and install the dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Training the Model
To train the model on your dataset:

1. Ensure your dataset is organized correctly under `root_dir`.
2. Run the following command:
    ```bash
    python main_train.py
    ```

The model will automatically save checkpoints after each epoch.

## Evaluating the Model
To evaluate the model performance on a set of test images, use:
```bash
python main_evaluate.py

## Results
After training, the following metrics will be displayed:

Training Accuracy: The accuracy during the training phase.
Validation Accuracy: The accuracy during validation after each epoch.
Test Accuracy: The overall accuracy when evaluating on unseen test data.

##Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.
Create a new feature branch (`git
