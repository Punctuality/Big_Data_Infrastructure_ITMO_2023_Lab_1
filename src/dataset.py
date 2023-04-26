from torch.utils.data import Dataset
import torch as t


class TokensDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = X.to(device)
        self.y = t.tensor(y).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]