import argparse
import configparser
import os

import train
import eval
from model import FakeNewsClassifier
import device_config

import torch as t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define config path')
    parser.add_argument('--config_path', type=str, help='path to INI config file', required=True)
    args = parser.parse_args()

    print(f"Workdir: {os.getcwd()}")


    config = configparser.ConfigParser()
    config.read(args.config_path)

    if config.get('system', 'device'):
        device = t.device(config['system']['device'])
    else:
        device = device_config.optimal_device()
    print(f"Device of choice: {device}")

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