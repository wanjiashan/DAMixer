import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.attention_matrix = None

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        h = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        self.attention_matrix = F.softmax(attn_energies, dim=1)
        return self.attention_matrix.unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        return self.v(energy).squeeze(2)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.tcn = nn.Sequential(
            nn.Conv1d(num_channels, 2 * num_channels, kernel_size=1),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=1),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=2),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=4),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=8),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=16),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=32),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, 2 * num_channels, kernel_size=kernel_size, dilation=64),
            nn.BatchNorm1d(2 * num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * num_channels, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(num_channels, output_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size)

    def forward(self, x):
        x, _ = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        attention_weights = self.attention(x[:, -1, :], x)
        x = torch.bmm(attention_weights, x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


input_size = 6
output_size = 1
num_channels = 64
kernel_size = 5
dropout = 0.2
hidden_size = 128
num_layers = 1
quantiles = [0.1, 0.5, 0.9]

model = TCN(input_size, output_size, num_channels, kernel_size, dropout)
criterion = QuantileLoss(quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)