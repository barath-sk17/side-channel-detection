import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
import torch.utils.data as data
import random
import torch.nn.functional as F
import shutil
from torch.optim.lr_scheduler import StepLR

# Define your functions and classes
def create_dataset(window, csv_folder):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    data = []
    # minlength = 100
    droplastrows = 8
    rowsperfile = 44 - droplastrows
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_folder, file))
            df.drop(df.tail(1).index,inplace=True)
            df.drop(df.head(1).index,inplace=True)
            df.replace('<not counted>', np.nan, inplace=True)
            df = df.dropna(axis=1, thresh=int(0.5 * len(df)))
            df = df.dropna(axis=0, how='any')
            df = df.drop(df.tail(8).index)
            rows = []
            df = df[['L1-icache-load-misses', 'L1-icache-loads']]

            for index, row in df.iterrows():
              rows.append((row.values).astype(int))
            for i in range(rowsperfile - window):
                feature = rows[i:i+window]
                target = rows[i+window]
                X.append(feature)
                y.append(target)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_eval_dataset(window, csv_folder):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    data = []
    droplastrows = 8
    rowsperfile = 44 - droplastrows
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_folder, file))
            df.drop(df.tail(1).index,inplace=True)
            df.drop(df.head(1).index,inplace=True)
            df.replace('<not counted>', np.nan, inplace=True)
            df = df.dropna(axis=1, thresh=int(0.5 * len(df)))
            df = df.dropna(axis=0, how='any')
            rows = []
            df = df[['L1-icache-load-misses', 'L1-icache-loads']]

            for index, row in df.iterrows():
              rows.append((row.values).astype(int))
            Xf = []
            yf = []
            for i in range(rowsperfile - window):
                feature = rows[i:i+window]
                target = rows[i+window]
                Xf.append(feature)
                yf.append(target)
            X.append(Xf)
            y.append(yf)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=40, num_layers=1, batch_first=True)
        self.linear = nn.Linear(40, 2)
        self.init_weights(self.linear)
        self.init_weights(self.lstm)

    def forward(self, x):

        batch = x.shape[0]
        sequence = x.shape[1]
        cols = x.shape[2]

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.linear(x)
        return x


    def init_weights(self, layer):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)




# Streamlit app code
def main():
    st.title("Training Side-Channel Data")

    # Sidebar options
    st.sidebar.header("Settings")
    window_size = st.sidebar.number_input("Window Size", min_value=1, value=20, step=1)
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=128, step=1)
    num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, value=120, step=1)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0, value=1e-4, step=1e-5)

    # Load data
    data_folder = 'data2'
    train_path = os.path.join(data_folder, 'train2')
    test_path = os.path.join(data_folder, 'test')
    X_train, y_train = create_dataset(window_size, train_path)
    X_test, y_test = create_dataset(window_size, test_path)

    # Training
    if st.button("Train Model"):
        st.text("Training in progress...")
        train_model(X_train, y_train, X_test, y_test, window_size, batch_size, num_epochs, learning_rate)

    if st.button("Model Evaluation"):
        model_eval(X_train, y_train, X_test, y_test, window_size, batch_size, num_epochs, learning_rate)
    
    if st.button("Transformer Model"):
        trans_model(X_train, y_train, X_test, y_test, batch_size, window_size)

def train_model(X_train, y_train, X_test, y_test, window_size, batch_size, num_epochs, learning_rate):
   
    Xtrain_max0 = torch.max(X_train[:,:,0]) # maximum column value between each data split
    Xtrain_max1 = torch.max(X_train[:,:,1])

    ytrain_max0 = torch.max(y_train[:,0]) # maximum column value between each data split
    ytrain_max1 = torch.max(y_train[:,1])


    Xtest_max0 = torch.max(X_test[:,:,0])
    Xtest_max1 = torch.max(X_test[:,:,1])

    ytest_max0 = torch.max(y_test[:,0])
    ytest_max1 = torch.max(y_test[:,1])

    max0 = max(Xtrain_max0, Xtest_max0, ytrain_max0, ytest_max0)
    max1 = max(Xtrain_max1, Xtest_max1, ytrain_max1, ytest_max1)


    X_train[:,:,0] /= max0
    X_train[:,:,1] /= max1

    X_test[:,:,0] /= max0
    X_test[:,:,1] /= max1

    y_train[:,0] /= max0
    y_train[:,1] /= max1

    y_test[:,0] /= max0
    y_test[:,1] /= max1

    # Create DataLoader
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size, drop_last=True)

    # Create RNN model
    rnn = RNNModel()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    for epoch in range(num_epochs):
        rnn.train()
        for X_batch, y_batch in loader:
            y_pred = rnn(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        rnn.eval()
        with torch.no_grad():
            y_pred_train = rnn(X_train)
            train_rmse = np.sqrt(criterion(y_pred_train, y_train))
            y_pred_test = rnn(X_test)
            test_rmse = np.sqrt(criterion(y_pred_test, y_test))

        st.text("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    

def model_eval(X_train, y_train, X_test, y_test, window_size, batch_size, num_epochs, learning_rate):
    st.title("Model Evaluation")

    Xtrain_max0 = torch.max(X_train[:,:,0]) # maximum column value between each data split
    Xtrain_max1 = torch.max(X_train[:,:,1])

    ytrain_max0 = torch.max(y_train[:,0]) # maximum column value between each data split
    ytrain_max1 = torch.max(y_train[:,1])


    Xtest_max0 = torch.max(X_test[:,:,0])
    Xtest_max1 = torch.max(X_test[:,:,1])

    ytest_max0 = torch.max(y_test[:,0])
    ytest_max1 = torch.max(y_test[:,1])

    max0 = max(Xtrain_max0, Xtest_max0, ytrain_max0, ytest_max0)
    max1 = max(Xtrain_max1, Xtest_max1, ytrain_max1, ytest_max1)
    
    X_train[:,:,0] /= max0
    X_train[:,:,1] /= max1

    X_test[:,:,0] /= max0
    X_test[:,:,1] /= max1

    y_train[:,0] /= max0
    y_train[:,1] /= max1

    y_test[:,0] /= max0
    y_test[:,1] /= max1

    # Create DataLoader
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size, drop_last=True)

    # Create RNN model
    rnn = RNNModel()

    sequence_length = window_size
    path = 'data2'
    X_vuln_eval, y_vuln_eval = create_eval_dataset(sequence_length, path + '/final/vuln')
    X_nonvuln_eval, y_nonvuln_eval = create_eval_dataset(sequence_length, path + '/final/nonvuln')

    X_nonvuln_eval[:,:,:,0] /= max0
    X_nonvuln_eval[:,:,:,1] /= max1

    X_vuln_eval[:,:,:,0] /= max0
    X_vuln_eval[:,:,:,1] /= max1

    y_nonvuln_eval[:,:,0] /= max0
    y_nonvuln_eval[:,:,1] /= max1

    y_vuln_eval[:,:,0] /= max0
    y_vuln_eval[:,:,1] /= max1

    # Calculate MSE for a single example
    st.header("MSE Calculation for a Single Example")
    tensor1 = rnn(X_nonvuln_eval[1,1,:,:].view(1, -1, 2))
    tensor2 = y_nonvuln_eval[1,1,:].view(1,2)
    mse = F.mse_loss(tensor1, tensor2)
    st.write("MSE:", mse)

    # Model Evaluation
    st.header("Model Evaluation")

    tn, tp, fn, fp = 0,0,0,0
    threshold = 6e-7
    percent = 1

    st.subheader("Non-vulnerable application detection")
    avgerror = 0
    for i in range(X_nonvuln_eval.shape[0]):
        flags = 0
        for j in range(X_nonvuln_eval.shape[1]):
            error = F.mse_loss(rnn(X_nonvuln_eval[i,j,:,:].unsqueeze(0)), y_nonvuln_eval[i,j,:].unsqueeze(0))
            avgerror += error
            if error >  threshold:
                flags += 1
        if flags >= (percent * X_nonvuln_eval.shape[1]):
            fp += 1
        else:
            tn += 1
    st.write("Average Error:", avgerror / (X_vuln_eval.shape[0] * X_vuln_eval.shape[1]))
    st.write("True Positive (TP):", tp)
    st.write("True Negative (TN):", tn)
    st.write("False Positive (FP):", fp)
    st.write("False Negative (FN):", fn)

    st.subheader("Vulnerable application detection")
    avgerror = 0
    for i in range(X_vuln_eval.shape[0]):
        flags = 0
        for j in range(X_vuln_eval.shape[1]):
            error = F.mse_loss(rnn(X_vuln_eval[i,j,:,:].unsqueeze(0)), y_vuln_eval[i,j,:].unsqueeze(0))
            avgerror += error
            if error > threshold:
                flags += 1
        if flags >= (percent * X_vuln_eval.shape[1]):
            tp += 1
        else:
            fn += 1

    st.write("Average Error:", avgerror / (X_vuln_eval.shape[0] * X_vuln_eval.shape[1]))
    st.write("True Positive (TP):", tp)
    st.write("True Negative (TN):", tn)
    st.write("False Positive (FP):", fp)
    st.write("False Negative (FN):", fn)


def trans_model(X_train, y_train, X_test, y_test, batch_size, window_size):
    
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size, drop_last=True)

    Xtrain_max0 = torch.max(X_train[:,:,0]) # maximum column value between each data split
    Xtrain_max1 = torch.max(X_train[:,:,1])

    ytrain_max0 = torch.max(y_train[:,0]) # maximum column value between each data split
    ytrain_max1 = torch.max(y_train[:,1])


    Xtest_max0 = torch.max(X_test[:,:,0])
    Xtest_max1 = torch.max(X_test[:,:,1])

    ytest_max0 = torch.max(y_test[:,0])
    ytest_max1 = torch.max(y_test[:,1])

    max0 = max(Xtrain_max0, Xtest_max0, ytrain_max0, ytest_max0)
    max1 = max(Xtrain_max1, Xtest_max1, ytrain_max1, ytest_max1)
    
    X_train[:,:,0] /= max0
    X_train[:,:,1] /= max1

    X_test[:,:,0] /= max0
    X_test[:,:,1] /= max1

    y_train[:,0] /= max0
    y_train[:,1] /= max1

    y_test[:,0] /= max0
    y_test[:,1] /= max1

    # Create DataLoader
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size, drop_last=True)

    # Create RNN model
    rnn = RNNModel()

    sequence_length = window_size
    path = 'data2'
    X_vuln_eval, y_vuln_eval = create_eval_dataset(sequence_length, path + '/final/vuln')
    X_nonvuln_eval, y_nonvuln_eval = create_eval_dataset(sequence_length, path + '/final/nonvuln')

    X_nonvuln_eval[:,:,:,0] /= max0
    X_nonvuln_eval[:,:,:,1] /= max1

    X_vuln_eval[:,:,:,0] /= max0
    X_vuln_eval[:,:,:,1] /= max1

    y_nonvuln_eval[:,:,0] /= max0
    y_nonvuln_eval[:,:,1] /= max1

    y_vuln_eval[:,:,0] /= max0
    y_vuln_eval[:,:,1] /= max1

    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            num_layers = 1
            input_size = 2
            # Replace LSTM with Transformer layer
            self.transformer = nn.Transformer(
                d_model=input_size,
                nhead=1,  # Number of heads in the multiheadattention models
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
            )
            self.linear = nn.Linear(2, 2)

        def forward(self, x):

            x = x.permute(1, 0, 2)
            x = self.transformer(x, x)
            x = x.permute(1, 0, 2)
            x = x[:, -1, :]

            x = self.linear(x)
            return x

    transformer = TransformerModel()

    learning_rate = 0.001
    num_epochs = 5

    # Loss and optimizer
    criterion2 = nn.MSELoss()
    optimizer2 = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        transformer.train()
        for X_batch, y_batch in loader:
            y_pred = transformer(X_batch)
            loss = criterion2(y_pred, y_batch)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
        # scheduler.step()
        transformer.eval()
        with torch.no_grad():
            y_pred = transformer(X_train)
            train_rmse = np.sqrt(criterion2(y_pred, y_train))
            y_pred = transformer(X_test)
            test_rmse = np.sqrt(criterion2(y_pred, y_test))
        st.text("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


        # Anomaly Detection Implementation 1
    tn, tp, fn, fp = 0,0,0,0
    threshold = 4.5e-4
    percent = 1
    
    st.subheader("Non-Vulnerable application detection")

    avgerror = 0
    for i in range(X_nonvuln_eval.shape[0]):
        flags = 0
        for j in range(X_nonvuln_eval.shape[1]):
            error = F.mse_loss(transformer(X_nonvuln_eval[i,j,:,:].view(1, X_nonvuln_eval.shape[2], X_nonvuln_eval.shape[3])), y_nonvuln_eval[i,j,:].view(1,X_nonvuln_eval.shape[3]))
            # print(error, " and " ,threshold)
            # print(error)
            avgerror += error
            if error >  threshold:
                flags += 1
        if flags >= (percent * X_nonvuln_eval.shape[1]):
            fp += 1
        else:
            tn += 1
            
    st.write("Average Error:", avgerror / (X_vuln_eval.shape[0] * X_vuln_eval.shape[1]))
    st.write("True Positive (TP):", tp)
    st.write("True Negative (TN):", tn)
    st.write("False Positive (FP):", fp)
    st.write("False Negative (FN):", fn)
    
    st.subheader("Vulnerable application detection")
    
    avgerror = 0
    for i in range(X_vuln_eval.shape[0]):
        flags = 0
        for j in range(X_vuln_eval.shape[1]):
            error = F.mse_loss(transformer(X_vuln_eval[i,j,:,:].view(1, X_vuln_eval.shape[2], X_nonvuln_eval.shape[3])), y_vuln_eval[i,j,:].view(1,X_nonvuln_eval.shape[3]))
            avgerror += error
            if error > threshold:
                flags += 1
    if flags >= (percent * X_vuln_eval.shape[1]):
        tp += 1
    else:
        fn += 1
        # print(flags)
    st.write("Average Error:", avgerror / (X_vuln_eval.shape[0] * X_vuln_eval.shape[1]))
    st.write("True Positive (TP):", tp)
    st.write("True Negative (TN):", tn)
    st.write("False Positive (FP):", fp)
    st.write("False Negative (FN):", fn)

if __name__ == "__main__":
    main()
