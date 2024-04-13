Certainly! Below is a GitHub README template based on the provided information:

---

# Side-channel Detection using LSTM/Transformer Model Time-Series Forecasting

## Overview

This project focuses on side-channel detection using time-series forecasting techniques, particularly employing LSTM (Long Short-Term Memory) and Transformer models. The data is collected using Intel PCM (Performance Counter Monitor), with a focus on L1 instruction cache (L1-icache) loads and load misses. However, other performance metrics can be incorporated as needed.

## Data Collection

Data collection involves obtaining various performance counters for feature selection. The primary metrics of interest are L1-icache loads and load misses. However, the framework allows for the inclusion of additional performance metrics as required.

## Data Pre-processing and Model Implementation

### Functions and Classes

- **create_dataset(window, csv_folder):** Function to transform time-series data from CSV files into a prediction dataset.
- **create_eval_dataset(window, csv_folder):** Similar to `create_dataset`, but for evaluation purposes.
- **RNNModel(nn.Module):** Defines a recurrent neural network (RNN) model using PyTorch's nn.Module class. It consists of an LSTM layer followed by a linear layer.
- **init_weights(layer):** Initializes the weights of the layers using Xavier uniform initialization for weights and zeros for biases.
- **train_model(), model_eval(), and trans_model():** Functions for training, evaluating, and transforming the data using the defined RNN model.

## Streamlit App

The project includes a Streamlit application for user interaction and visualization.

### Features:

- Sidebar options for setting parameters such as window size, batch size, number of epochs, and learning rate.
- Buttons to trigger the training, model evaluation, and transformer model processes.
- Results and metrics displayed using Streamlit's UI components.

## Training Process

1. Data is loaded from CSV files.
2. Input and target data are normalized.
3. Data loaders for training are created.
4. The RNN model, loss function (MSELoss), and optimizer (Adam) are initialized.
5. The model is trained using the specified number of epochs, with parameters updated based on backpropagation.
6. Train and test RMSE (Root Mean Squared Error) are printed for each epoch.

## Model Evaluation

- The trained model is evaluated on test data.
- Metrics such as True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN) are calculated for both vulnerable and non-vulnerable applications.
- Anomaly detection based on Mean Squared Error (MSE) is performed.

## Transformer Model

- Defines a transformer model similar to the RNN model.
- Initializes the transformer model, loss function, and optimizer.
- Trains the transformer model and evaluates it on non-vulnerable and vulnerable application data.
- Calculates and displays metrics similar to the RNN model evaluation.

## Usage

1. Clone the repository.
2. Install the required dependencies (`requirements.txt`).
3. Run the Streamlit app using `streamlit run app.py`.
4. Interact with the app to set parameters and perform training, evaluation, and transformation tasks.

