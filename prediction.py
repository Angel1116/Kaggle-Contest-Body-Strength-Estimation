import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import random
from sklearn.preprocessing import MinMaxScaler

training_dataroot = 'lab1_basic_training.csv' 
testing_dataroot = 'lab1_basic_testing.csv'   
output_dataroot = 'lab1_basic.csv'

training_datalist =  [] 
testing_datalist =  [] 
output_datalist =  [] 


with open(training_dataroot, newline='') as csvfile:
  training_datalist = pd.read_csv(training_dataroot).to_numpy()

with open(testing_dataroot, newline='') as csvfile:
  testing_datalist = pd.read_csv(testing_dataroot).to_numpy()
  
def SplitData(data, split_ratio):
    """
    Splits the given dataset into training and validation sets based on the specified split ratio.

    Parameters:
    - data (numpy.ndarray): The dataset to be split. It is expected to be a 2D array where each row represents a data point and each column represents a feature.
    - split_ratio (float): The ratio of the data to be used for training. For example, a value of 0.8 means 80% of the data will be used for training and the remaining 20% for validation.

    Returns:
    - training_data (numpy.ndarray): The portion of the dataset used for training.
    - validation_data (numpy.ndarray): The portion of the dataset used for validation.

    """
    training_data = []
    validation_data = []
    
    # TODO
    # Shuffle the data
    np.random.shuffle(data) # make the data index random

    # Determine the split point
    split_index = int(len(data) * split_ratio)
    
     # Split the data into training and validation sets
    training_data = data[:split_index]
    validation_data = data[split_index:]

    return training_data, validation_data

def PreprocessData(data):
    """
    Preprocess the given dataset and return the result.

    Parameters:
    - data (numpy.ndarray): The dataset to preprocess. It is expected to be a 2D array where each row represents a data point and each column represents a feature.

    Returns:
    - preprocessedData (numpy.ndarray): Preprocessed data.
    """
    preprocessedData = []

    # Remove rows with missing data (NaN)
    data = data[~np.isnan(data).any(axis=1)]

    # Replace outliers using IQR
    # Initialize preprocessedData with a copy of the data
    preprocessedData = np.copy(data)
    
    for col in range(data.shape[1]):
        # Get the 25th and 75th percentiles (Q1 and Q3)
        Q1 = np.percentile(data[:, col], 25)
        Q3 = np.percentile(data[:, col], 75)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect outliers
        outliers_mask = (data[:, col] < lower_bound) | (data[:, col] > upper_bound)
        
        # Replace outliers with the column's median (you can choose mean or other values)
        col_median = np.median(data[:, col][~outliers_mask])
        preprocessedData[outliers_mask, col] = col_median


    return preprocessedData

def Remove_noise(data):
    def remove_for_x(x, data):
        h = 1  # Replace with the desired value of h
        
        # 1. Select data points within the range (x-h, x+h)
        mask = (X > (x - h)) & (X < (x + h))
        X_in_range = X[mask]
        y_in_range = y[mask]
        
        # 2. Compute the mean and standard deviation for the selected range
        mean_y = np.mean(y_in_range)
        std_y = np.std(y_in_range)
        
        # 3. Identify and remove outliers beyond 2 standard deviations
        threshold = 2  # Adjust the threshold as needed
        mask_outliers = np.abs(y_in_range - mean_y) < threshold * std_y
        
        # Retain data points that are not outliers
        X_filtered = X_in_range[mask_outliers]
        y_filtered = y_in_range[mask_outliers]
        
        # 4. Update cleaned data by retaining points outside (x-h, x+h)
        mask_out_of_range = ~mask
        X_remaining = X[mask_out_of_range]
        y_remaining = y[mask_out_of_range]
        X_cleaned = np.concatenate([X_remaining, X_filtered])
        y_cleaned = np.concatenate([y_remaining, y_filtered])
        data = np.column_stack((X_cleaned, y_cleaned))
        return data

    
    for x in range(40,110,1):
        X = data[:, 0]
        y = data[:, 1]
        data = remove_for_x(x,data)
        
    return data


def Regression(dataset):
    """
    Performs regression on the given dataset and return the coefficients.

    Parameters:
    - dataset (numpy.ndarray): A 2D array where each row represents a data point.

    Returns:
    - w (numpy.ndarray): The coefficients of the regression model. For example, y = w[0] + w[1] * x + w[2] * x^2 + ...
    """

    X = dataset[:, :1]
    y = dataset[:, 1]

    # Decide on the degree of the polynomial
    degree = 1 # For example, quadratic regression

    X_reshaped = X.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = scaler.fit_transform(X_reshaped)

    # Add polynomial features to X
    X_poly = np.ones((X_normalized.shape[0], 1))  # Add intercept term (column of ones)
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X_normalized ** d))  # Add x^d terms to feature matrix

    # Initialize coefficients (weights) to small random values
    num_dimensions = X_poly.shape[1]  # Number of features (including intercept and polynomial terms)
    w_change = np.random.randn(num_dimensions) * 0.01

    # Set hyperparameters
    num_iteration = 150000 
    learning_rate = 0.01 

    y_reshaped = y.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_y_reshaped = scaler.fit_transform(y_reshaped)
    y_normalized = scaled_y_reshaped.reshape(-1)

    # Gradient Descent
    m = len(y_normalized)  # Number of data points
    min_cost = 100
    w = np.random.randn(num_dimensions) * 0.01
    for iteration in range(num_iteration):
        # Prediction using current weights and compute error
        y_pred = np.dot(X_poly, w_change)
        error = y_pred - y_normalized

        # Compute gradient
        gradient = (1/m) * np.dot(X_poly.T, error)

        # Update the weights
        w_change -= learning_rate * gradient

        # Optionally, print the cost every 10000 iterations
        cost = (1/(2*m)) * np.sum(error**2)
        if cost < min_cost:
            min_cost = cost
            w = w_change
        if iteration % 10000 == 0:
            print(f"iteration:{iteration}, cost:{cost}")
            
    # Denormalize the weights to match the original scale
    print(f"minimum_cost:{cost},{w}")
    w_denorm = np.zeros_like(w)
    X_min = np.min(X)  # Original X_min used for scaling
    X_max = np.max(X)  # Original X_max used for scaling
    y_min = np.min(y)  # Original y_min used for scaling
    y_max = np.max(y) 
    X_range = X_max - X_min
    y_range = y_max - y_min
    if degree == 1:
        # Linear model case
        w_denorm[0] = w[0] * y_range + y_min - (w[1] * X_min * y_range) / X_range
        w_denorm[1] = w[1] * y_range / X_range
    elif degree == 2:
        # Quadratic model case
        # Note: Quadratic models require more complex denormalization
        w_denorm[0] = w[0] * y_range + y_min - (w[1] * X_min * y_range) / X_range - (w[2] * (X_min ** 2) * y_range) / (X_range ** 2)
        w_denorm[1] = w[1] * y_range / X_range
        w_denorm[2] = w[2] * y_range / (X_range ** 2)
    elif degree == 3:
        # Cubic model case
        # Note: Cubic models require even more complex denormalization
        w_denorm[0] = (
            w[0] * y_range
            + y_min
            - (w[1] * X_min * y_range) / X_range
            - (w[2] * (X_min ** 2) * y_range) / (X_range ** 2)
            - (w[3] * (X_min ** 3) * y_range) / (X_range ** 3)
        )
        w_denorm[1] = w[1] * y_range / X_range
        w_denorm[2] = w[2] * y_range / (X_range ** 2)
        w_denorm[3] = w[3] * y_range / (X_range ** 3)
    elif degree == 4:
        # Quartic model case (四次多項式)
        # Note: Quartic models require even more complex denormalization
        w_denorm[0] = (
            w[0] * y_range
            + y_min
            - (w[1] * X_min * y_range) / X_range
            - (w[2] * (X_min ** 2) * y_range) / (X_range ** 2)
            - (w[3] * (X_min ** 3) * y_range) / (X_range ** 3)
            - (w[4] * (X_min ** 4) * y_range) / (X_range ** 4)
        )
        w_denorm[1] = w[1] * y_range / X_range
        w_denorm[2] = w[2] * y_range / (X_range ** 2)
        w_denorm[3] = w[3] * y_range / (X_range ** 3)
        w_denorm[4] = w[4] * y_range / (X_range ** 4)

    return w_denorm, X_normalized, y_normalized


def MakePrediction(w, test_dataset):
    """
    Predicts the output for a given test dataset using a regression model.

    Parameters:
    - w (numpy.ndarray): The coefficients of the model, where each element corresponds to
                               a coefficient for the respective power of the independent variable.
    - test_dataset (numpy.ndarray): A 1D array containing the input values (independent variable)
                                          for which predictions are to be made.

    Returns:
    - list/numpy.ndarray: A list or 1d array of predicted values corresponding to each input value in the test dataset.
    """
    prediction = []
    
    # Add polynomial features to the test dataset
    degree = len(w) - 1  # The degree of the polynomial model is determined by the length of w
    X_poly = np.ones((test_dataset.shape[0], 1))  # Start with the intercept term (column of ones)

    # Add columns corresponding to x, x^2, ..., x^degree
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, test_dataset.reshape(-1, 1) ** d))

    # Make predictions using matrix multiplication (dot product)
    prediction = X_poly.dot(w)
    
    return prediction


# (1) Split data
training_data, validation_data = SplitData(training_datalist, 0.90)

# (2) Preprocess data
training_data_cleaned0 = PreprocessData(training_data)
validation_data_cleaned = PreprocessData(validation_data)
training_data_cleaned = Remove_noise(training_data_cleaned0)

validation_X = validation_data_cleaned[:, 0]
validation_y = validation_data_cleaned[:, 1]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(validation_X, validation_y, color='blue', alpha=0.5, edgecolor='k')
plt.title('Scatter Plot of Validation Data')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.grid(True)
plt.show()

df_cleaned = pd.DataFrame(validation_data_cleaned, columns=['X', 'y'])
file_path = 'validation_data_cleaned_new.xlsx'
df_cleaned.to_excel(file_path, index=False)
print(f"Data has been saved to {file_path}")

# (3) Train regression model
w, X_normalized, y_normalized = Regression(training_data_cleaned)

# (4) Predict validation dataset's answer, calculate MAPE comparing to the ground truth
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
validation_X = validation_data_cleaned[:, 0:1]  # Input (independent variable)
validation_y = validation_data_cleaned[:, 1]    # Ground truth (dependent variable)
validation_predict = MakePrediction(w, validation_X)
validation_mape = MAPE(validation_y, validation_predict)
print(f"Validation MAPE: {validation_mape:.2f}%")

# (5) Make prediction of testing dataset and store the values in output_datalist
test_X = testing_datalist[:, 0:1]  # Extract test input (X)
output_datalist = MakePrediction(w, test_X) # Make predictions using the trained model

# Assume that output_datalist is a list (or 1d array) with length = 100

with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['Id', 'gripForce'])
  for i in range(len(output_datalist)):
    writer.writerow([i,output_datalist[i]])


