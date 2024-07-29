# friendly-octo-spork

## Potato Leaf Disease Classification

This project aims to classify potato leaf diseases using images from the Kaggle Plant Village dataset. The goal is to build a machine learning model that can accurately identify different types of potato leaf diseases to assist in early detection and treatment.

*Table of Contents*
Introduction
Dataset
Requirements
Project Structure
Installation
Usage
Model Training
Results
Contributing
License
Introduction

Potato crops are susceptible to various diseases that can significantly reduce yield and quality. Early detection and identification of these diseases are crucial for effective management and control. This project leverages deep learning techniques to classify potato leaf diseases using images from the Plant Village dataset.

Dataset

The dataset used in this project is the Plant Village dataset, which is available on Kaggle. It contains thousands of images of healthy and diseased plant leaves, including potato leaves. The dataset is divided into the following classes:

Healthy
Early Blight
Late Blight
You can download the dataset from Kaggle and place it in the data/ directory.

Requirements

Python 3.6+
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Scikit-learn
OpenCV
Project Structure

bash
Copy code
potato-leaf-disease-classification/
│
├── data/
│   ├── train/
│   ├── test/
│   └── validation/
├── notebooks/
│   ├── potato_disease_classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt



Model Training

The model is trained using a Convolutional Neural Network (CNN) architecture. You can find the model definition and training process includes Loading and preprocessing the images.
Defining the CNN architecture.
Training the model using the training dataset.
Validating the model using the validation dataset.
Results

The trained model achieves an accuracy of 98% on the test dataset. Detailed evaluation metrics and visualizations can be found in the notebook.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.


