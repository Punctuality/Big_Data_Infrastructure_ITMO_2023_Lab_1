import torch.nn as nn
import torch as t


class FakeNewsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # , padding_idx=0)
        self.dropout_1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, 64)
        self.dropout_3 = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_1(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout_2(x)
        x = t.relu(self.dense(x))
        x = self.dropout_3(x)
        x = t.sigmoid(self.out(x))
        return x


def save_model(model, path):
    t.save(model.state_dict(), path)