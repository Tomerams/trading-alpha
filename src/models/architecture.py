import torch
import torch.nn as nn
from pytorch_tcn import TCN

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=2, nhead=4):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x)
        h = self.transformer(h)
        return self.fc(h[:, -1, :])

class TransformerRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_heads=4, output_size=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.embedding(x)
        attn_out, _ = self.self_attn(h, h, h)
        h = h + attn_out
        h = h.permute(1, 0, 2)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super().__init__()
        self.tcn = TCN(
            input_size,
            [hidden_size] * 3,
            kernel_size=3,
            dropout=0.1,
            causal=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = h[:, :, -1]
        return self.fc(h)

    @property
    def model(self):
        return self

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    @property
    def model(self):
        return self

class TransformerTCN(nn.Module):
    def __init__(self, input_dim, tcn_channels, transformer_dim, n_heads, n_layers, output_dim):
        super().__init__()
        self.tcn = TCN(
            input_dim,
            tcn_channels,
            kernel_size=3,
            dropout=0.1,
            causal=True
        )
        self.proj = nn.Linear(tcn_channels[-1], transformer_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(transformer_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = h.transpose(1, 2)
        h = self.proj(h)
        h = self.transformer(h)
        return self.head(h[:, -1, :])

class TransformerTCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = TransformerTCN(
            input_dim=input_size,
            tcn_channels=[hidden_size] * 3,
            transformer_dim=hidden_size,
            n_heads=4,
            n_layers=2,
            output_dim=output_size
        )

    def forward(self, x):
        return self.net(x)
