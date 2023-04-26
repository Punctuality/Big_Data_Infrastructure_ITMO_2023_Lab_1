import pandas as pd
import torch as t
from torchtext.vocab import Vocab

import corpus
import preprocessing
from model import FakeNewsClassifier


def eval_model_on_test(model: FakeNewsClassifier, device: t.device, config: dict, vocab: Vocab) -> None:

    test_path = config['paths']['test_path']
    max_pad_len = int(config['preprocessing']['max_pad_len'])
    path_to_save = config['paths']['submission_path']

    test_data = corpus.read_dataframe(test_path)
    test_corpus = corpus.load_corpus(test_data)
    test_tokens = preprocessing.tokenize(test_corpus)
    voc_test_tokens = preprocessing.vocab_tokens(vocab, test_tokens)
    padded_test_tokens = preprocessing.padding_indexes(voc_test_tokens, max_pad_len)

    model.eval()
    model = model.to(device)
    padded_test_tokens = padded_test_tokens.to(device)
    with t.no_grad():
        y_pred = model(padded_test_tokens)

    test_data["label"] = list(map(lambda x: 1 if x > 0.5 else 0, y_pred.to("cpu").numpy()))
    test_data.to_csv(path_to_save, index=True)