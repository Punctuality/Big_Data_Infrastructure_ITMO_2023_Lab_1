from typing import Union

import pandas as pd
import torch as t
from torch import optim
from torch.utils.data import DataLoader
import time

from torchtext.vocab import Vocab

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
    df_train: pd.DataFrame
    x_val: t.Tensor
    df_val: pd.DataFrame

    y_train: t.Tensor
    y_val: t.Tensor

    vocab: Vocab

    def __init__(self, x_train, x_val, df_train, df_val, y_train, y_val, vocab):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.df_train = df_train
        self.df_val = df_val
        self.vocab = vocab

def train_baseline(
        config: dict, \
        device: t.device, \
) -> tuple[FakeNewsClassifier, TrainingData]:

    # Load all data and preprocess it
    print("Loading data")

    train_path = config['paths']['train_path']
    train_data = corpus.read_dataframe(train_path)

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        train_data.drop('label', axis=1), train_data['label'],
        test_size=float(config['training']['train_test_ratio']),
        random_state=int(config['training']['seed'])
    )

    print("Extracting corpus")

    train_corpus = corpus.load_corpus(x_train)
    val_corpus = corpus.load_corpus(x_val)

    print("Tokenizing corpus")
    train_tokens = preprocessing.tokenize(train_corpus)
    val_tokens = preprocessing.tokenize(val_corpus)

    print("Building vocab")
    vocab = preprocessing.build_vocab(train_tokens + val_tokens, int(config['preprocessing']['vocab_size']))

    voc_tokens = preprocessing.vocab_tokens(vocab, train_tokens)
    voc_tokens_val = preprocessing.vocab_tokens(vocab, val_tokens)

    print("Padding embeddings")
    max_pad_len = int(config['preprocessing']['max_pad_len'])
    padded_tokens = preprocessing.padding_indexes(voc_tokens, max_pad_len)
    padded_tokens_val = preprocessing.padding_indexes(voc_tokens_val, max_pad_len)

    # Setup model

    embedding_dim = int(config['model']['embedding_dim'])
    hidden_dim = int(config['model']['hidden_dim'])
    num_layers = int(config['model']['n_layers'])
    dropout = float(config['model']['dropout'])
    model = FakeNewsClassifier(len(vocab), embedding_dim, hidden_dim, num_layers, dropout).to(device)

    batch_size = 64
    train_dataset = dataset.TokensDataset(padded_tokens, y_train.values, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = dataset.TokensDataset(padded_tokens_val, y_val.values, device)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss = nn.BCELoss()
    lr = float(config['training']['lr'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training")
    epochs = int(config['training']['epochs'])
    measure_time(lambda: train(model, train_dataloader, val_dataloader, loss, optimizer, epochs), "training")()
    model.eval()

    save_path = config['paths']['model_path']
    if save_path:
        save_model(model, save_path)

    vocab_path = config['paths']['vocab_path']
    t.save(vocab, vocab_path)

    return model, TrainingData(
        padded_tokens, padded_tokens_val,
        x_train, x_val,
        t.tensor(y_train.values), t.tensor(y_val.values),
        vocab
    )



