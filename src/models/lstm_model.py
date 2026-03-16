import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import yaml
import os

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the output from the last time step
        return out

class LSTMModel:
    def __init__(self, input_size, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        lstm_cfg = self.config["models"]["lstm"]
        self.hidden_size = lstm_cfg["hidden_size"]
        self.num_layers = lstm_cfg["num_layers"]
        self.dropout = lstm_cfg["dropout"]
        self.lr = lstm_cfg["learning_rate"]
        self.epochs = lstm_cfg["epochs"]
        self.batch_size = lstm_cfg["batch_size"]
        self.sequence_length = self.config["data"]["sequence_length"]
        self.input_size = input_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = LSTMNetwork(input_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model_path = os.path.join(self.config["paths"]["models_dir"], "lstm_model.pth")

    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Create sequences of length `sequence_length`."""
        xs, ys = [], []
        for i in range(len(X) - self.sequence_length):
            xs.append(X[i:(i + self.sequence_length)])
            if y is not None:
                ys.append(y[i + self.sequence_length])
        
        if y is not None:
            return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)
        return torch.tensor(np.array(xs), dtype=torch.float32)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_seq, y_seq = self.create_sequences(X_train.values, y_train.values)
        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Print every epoch for better visibility
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(loader):.4f}")
                
        self.save_model()

    def predict(self, X: pd.DataFrame) -> list:
        # Prepend logic for inference is needed to have full sequence length window.
        # This function expects `X` to already have shape (batch_size, sequence_length, features)
        # or it handles sequence creation for the batch.
        
        if len(X) <= self.sequence_length:
            raise ValueError(f"Input data length must be greater than sequence length {self.sequence_length}")
            
        X_seq = self.create_sequences(X.values)
        X_seq = X_seq.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_seq)
        
        return outputs.cpu().numpy().flatten().tolist()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
