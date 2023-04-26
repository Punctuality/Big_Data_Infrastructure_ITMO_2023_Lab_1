import configparser
import os

import train
import eval
from model import FakeNewsClassifier
from src import device_config

import pickle as pk
import torch as t

if __name__ == "__main__":
    print(f"Workdir: {os.getcwd()}")

    device = device_config.optimal_device()
    print(f"Device of choice: {device}")

    config = configparser.ConfigParser()
    config.read('../config.ini')

    model_path = config['paths']['model_path']
    vocab_path = config['paths']['vocab_path']

    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print("Model and vocab is present")
        vocab = t.load(vocab_path)

        embedding_dim = int(config['model']['embedding_dim'])
        hidden_dim = int(config['model']['hidden_dim'])
        num_layers = int(config['model']['n_layers'])
        dropout = float(config['model']['dropout'])

        model = FakeNewsClassifier(len(vocab), embedding_dim, hidden_dim, num_layers, dropout).to(device)
        model.load_state_dict(t.load(model_path))

    else:
        model, data = train.train_baseline(config, device)
        vocab = data.vocab

    print("Evaluating model")
    eval.eval_model_on_test(model, device, config, vocab)