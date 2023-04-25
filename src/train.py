from typing import Union

import torch as t
from torch import optim
from torch.utils.data import DataLoader
import time

import corpus
import preprocessing
from model import *
import dataset
import device_config


def accuracy(model, test_dataloader):
    model.eval()
    with t.no_grad():
        sum_acc = 0
        for batch in test_dataloader:
            X, y = batch
            y_pred = model(X)
            y_pred = y_pred.squeeze()
            sum_acc += t.sum((y_pred > 0.5) == y)
        return sum_acc / (len(test_dataloader) * test_dataloader.batch_size)


def train(model, dataloader, val_dataloader, loss, optimizer, epochs):
    print(f"Staring acc_train: {accuracy(model, dataloader)} acc_val: {accuracy(model, val_dataloader)}")

    val_acc = 0
    for epoch in range(epochs):
        model.train()
        l = 0
        for batch in dataloader:
            X, y = batch
            optimizer.zero_grad()
            y_pred = model(X)
            y_pred = y_pred.squeeze()
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

        train_acc = accuracy(model, dataloader)
        val_acc = accuracy(model, val_dataloader)

        print(f"Epoch {epoch + 1} loss: {l.item()} acc_train: {train_acc} acc_val: {val_acc}")

    print(f"Final accuracy: {val_acc}")


def measure_time(callable, name):
    def wrapper():
        start = time.time()
        result = callable()
        end = time.time()
        print(f"Time elapsed ({name}): {end - start}")
        return result

    return wrapper


class TrainingData:
    x_train: t.Tensor
    x_val: t.Tensor
    x_test: t.Tensor

    y_train: t.Tensor
    y_val: t.Tensor

    def __init__(self, x_train, x_val, x_test, y_train, y_val):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val

def train_baseline(train_path: str, test_path: str, save_path: Union[None, str] = None) -> tuple[FakeNewsClassifier, TrainingData]:
    # Load all data and preprocess it
    print("Loading data")

    train_data = corpus.read_dataframe(train_path)
    test_data = corpus.read_dataframe(test_path)

    print("Extracting corpus")
    train_corpus = corpus.load_corpus(train_data.drop('label', axis=1))
    test_corpus = corpus.load_corpus(test_data)

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        train_corpus, train_data['label'], test_size=0.2, random_state=42
    )

    x_train = x_train.copy()
    x_val = x_val.copy()

    print("Tokenizing corpus")
    train_tokens = preprocessing.tokenize(x_train)
    val_tokens = preprocessing.tokenize(x_val)
    test_tokens = preprocessing.tokenize(test_corpus)

    print("Building vocab")
    vocab = preprocessing.build_vocab(train_tokens + val_tokens, 5000)

    voc_tokens = preprocessing.vocab_tokens(vocab, train_tokens)
    voc_tokens_val = preprocessing.vocab_tokens(vocab, val_tokens)
    voc_tokens_test = preprocessing.vocab_tokens(vocab, test_tokens)

    print("Padding embeddings")
    padded_tokens = preprocessing.padding_indexes(voc_tokens, 25)
    padded_tokens_val = preprocessing.padding_indexes(voc_tokens_val, 25)
    padded_tokens_test = preprocessing.padding_indexes(voc_tokens_test, 25)

    # Setup model

    device = device_config.optimal_device()
    print(f"Device of choice: {device}")

    model = FakeNewsClassifier(len(vocab), 40, 100, 2, 0.3).to(device)

    batch_size = 64
    train_dataset = dataset.TokensDataset(padded_tokens, y_train.values, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = dataset.TokensDataset(padded_tokens_val, y_val.values, device)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training")
    measure_time(lambda: train(model, train_dataloader, val_dataloader, loss, optimizer, 5), "training")()
    model.eval()

    if save_path:
        save_model(model, save_path)

    return model, TrainingData(
        padded_tokens, padded_tokens_val, padded_tokens_test, t.tensor(y_train.values), t.tensor(y_val.values)
    )



