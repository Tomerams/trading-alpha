import torch
import torch.nn as nn
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, InputLayer

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)  # Output 3 values

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=2, nhead=4):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class TransformerRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_heads=4, output_size=3, dropout=0.1):
        super(TransformerRNNModel, self).__init__()
        
        self.embedding = nn.Linear(input_size, hidden_size)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)  # Predicting relative price changes
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.embedding(x)
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = x.permute(1, 0, 2)
        
        transformer_out = self.transformer(x)
        transformer_out = transformer_out.permute(1, 0, 2)
        
        lstm_out, _ = self.lstm(transformer_out)
        out = self.fc(lstm_out[:, -1, :])
        
        return out


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TCNModel:
    def __init__(self, input_size, hidden_size, output_size=3):
        self.model = Sequential(
            [
                InputLayer(input_shape=(input_size, 1)),
                TCN(
                    nb_filters=hidden_size,
                    kernel_size=3,
                    dilations=[1, 2, 4, 8],
                    return_sequences=False,
                ),
                Dense(output_size, activation="linear"),  # Output 3 values
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X_train, y_train, epochs=20, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)


class GRUModel:
    def __init__(self, input_size, hidden_size, output_size=3):
        self.model = Sequential(
            [
                Input(shape=(input_size, 1)),
                GRU(hidden_size, return_sequences=True),
                Dropout(0.2),
                GRU(hidden_size),
                Dense(output_size, activation="linear"),  # Output 3 values
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X_train, y_train, epochs=20, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)
