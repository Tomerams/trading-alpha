import torch
import torch.nn as nn
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, InputLayer

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure proper shape for LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, nhead=4):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, batch_first=True
            ),  # ✅ Ensure batch_first
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure correct shape (batch, seq_len, features)
        x = self.encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # ✅ Select the last timestep correctly


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNLSTMModel, self).__init__()

        # CNN Layers
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:  # Ensuring proper dimensions
            x = x.unsqueeze(1)

        x = x.permute(0, 2, 1)  # Reshaping for CNN: [batch, features, sequence]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = x.permute(0, 2, 1)  # Reshape for LSTM: [batch, sequence, features]
        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out[:, -1, :])  # Last timestep output


class TCNModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential([
            InputLayer(input_shape=(input_size, 1)),  # Explicitly use InputLayer
            TCN(nb_filters=hidden_size, kernel_size=3, dilations=[1, 2, 4, 8], return_sequences=False),
            Dense(output_size, activation="linear")
        ])
        
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X_train, y_train, epochs=20, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)



class GRUModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential(
            [
                Input(shape=(input_size, 1)),
                GRU(hidden_size, return_sequences=True),
                Dropout(0.2),
                GRU(hidden_size),
                Dense(output_size, activation="linear"),
            ]
        )

        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X_train, y_train, epochs=20, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)
