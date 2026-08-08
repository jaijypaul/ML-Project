# Iris Species Classification using PyTorch

This notebook demonstrates how to build, train, and evaluate a simple Artificial Neural Network (ANN) using PyTorch to classify Iris flower species based on their sepal and petal measurements.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Definition](#model-definition)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Making Predictions](#making-predictions)
- [Saving and Loading the Model](#saving-and-loading-the-model)

## Introduction
The Iris dataset is a classic and very easy multi-class classification dataset. The dataset consists of 150 samples from three species of Iris flowers (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

This notebook walks through the process of:
1.  Loading the Iris dataset.
2.  Preprocessing the data for neural network input.
3.  Defining a simple feedforward neural network using `torch.nn`.
4.  Training the network.
5.  Evaluating its performance.
6.  Demonstrating how to make new predictions.
7.  Saving and loading the trained model.

## Setup
To run this notebook, you'll need the following libraries:
- `torch` (PyTorch)
- `pandas`
- `matplotlib`
- `scikit-learn`

These are typically installed in a Google Colab environment. If running locally, you can install them via pip:
```bash
pip install torch pandas matplotlib scikit-learn
Data Loading and Preprocessing
Load Data: The Iris dataset is loaded directly from a CSV file hosted on Gist into a pandas DataFrame.
Encode Labels: The categorical species names (Setosa, Versicolor, Virginica) are converted into numerical labels (0.0, 1.0, 2.0).
Feature-Target Split: The dataset is split into features (X) and target (y).
Train-Test Split: The data is further split into training and testing sets (80% training, 20% testing) to evaluate the model's generalization performance.
Tensor Conversion: NumPy arrays are converted into PyTorch FloatTensor for features and LongTensor for target labels, as required by PyTorch for neural network operations.
Model Definition
A simple feedforward neural network (Model class) is defined using torch.nn.Module:

Input Layer: 4 input features (sepal length, sepal width, petal length, petal width).
Hidden Layers: Two hidden layers, each with 8 neurons, using ReLU activation functions.
Output Layer: 3 output neurons, corresponding to the three Iris species.
class Model(nn.Module):
  def __init__(self,in_features=4,h1=8,h2=8,out_features=3):
    super().__init__()
    self.fc1=nn.Linear(in_features,h1)
    self.fc2=nn.Linear(h1,h2)
    self.out=nn.Linear(h2,out_features)

  def forward(self,x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=self.out(x)
    return x
An instance of this model is created and initialized with a manual seed for reproducibility.

Training the Model
Loss Function: nn.CrossEntropyLoss() is used, which is suitable for multi-class classification problems with integer labels.
Optimizer: The Adam optimizer (torch.optim.Adam) is chosen with a learning rate of 0.01.
Training Loop: The model is trained for 200 epochs. In each epoch:
A forward pass is performed to get predictions.
The loss is calculated.
Gradients are zeroed, backpropagation is performed, and optimizer steps are taken to update model weights.
Loss is printed every 10 epochs.
Evaluation
After training, the model's performance is evaluated on the test set:

The torch.no_grad() context is used to disable gradient calculations, as they are not needed for evaluation and save memory.
Predictions are made on the X_test data.
The test loss is calculated.
An accuracy check is performed by iterating through the test data, comparing the model's predicted class (argmax of output) with the true label.
The number of correctly classified samples is printed.

Making Predictions
To predict the species of a new Iris flower, a new tensor with 4 features is created. The trained model then makes a prediction, and the argmax() function is used to determine the predicted class label. This label is then mapped back to the species name.

new_iris_data= torch.tensor([4.7,2.8,1.3,0.5]) # Example new data
with torch.no_grad():
  eval=model(new_iris_data)
  predicted_class_id = eval.argmax().item()
  # Map predicted_class_id to species name (e.g., 'setosa', 'Versicolor', 'Virginica')
  print(predicted_species_name)
Saving and Loading the Model
Saving: The trained model's state dictionary (containing all learned parameters) is saved to a file named iris_data_model using torch.save(model.state_dict(), 'iris_data_model').
Loading: A new instance of the Model class is created, and the saved state dictionary is loaded into it using new_model.load_state_dict(torch.load('iris_data_model')), allowing the model to be reused without retraining.
