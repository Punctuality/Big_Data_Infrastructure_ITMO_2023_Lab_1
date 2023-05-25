import torch as t
import numpy as np

from src.preprocessing import *


class TestPreprocessingSpec:

    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one."
    ]

    def test_tokenize(self):
        tokens = tokenize(self.corpus)
        assert len(tokens) == 3
        assert tokens[0] == ['this', 'is', 'the', 'first', 'document', '.']
        assert tokens[1] == ['this', 'document', 'is', 'the', 'second', 'document', '.']
        assert tokens[2] == ['and', 'this', 'is', 'the', 'third', 'one', '.']

    def test_build_vocab(self):
        tokens = tokenize(self.corpus)
        vocab = build_vocab(tokens, 10)
        assert len(vocab) == 10
        assert [vocab[token] for token in [
            '<unk>', '.', 'document', 'is', 'the', 'this', 'and', 'first', 'one', 'second'
            ]] == list(range(10))

    def test_vocab_tokens(self):
        tokens = tokenize(self.corpus)
        vocab = build_vocab(tokens, 10)
        vb_tokens = vocab_tokens(vocab, tokens)
        assert len(vb_tokens) == 3
        assert vb_tokens[0].tolist() == [5, 3, 4, 7, 2, 1]
        assert vb_tokens[1].tolist() == [5, 2, 3, 4, 9, 2, 1]
        assert vb_tokens[2].tolist() == [6, 5, 3, 4, 0, 8, 1]

    def test_padding_indexes(self):
        tokens = tokenize(self.corpus)
        vocab = build_vocab(tokens, 10)
        vb_tokens = vocab_tokens(vocab, tokens)
        padded = padding_indexes(vb_tokens, 10)
        assert len(padded) == 3
        assert padded[0].tolist() == [0, 0, 0, 0, 5, 3, 4, 7, 2, 1]
        assert padded[1].tolist() == [0, 0, 0, 5, 2, 3, 4, 9, 2, 1]
        assert padded[2].tolist() == [0, 0, 0, 6, 5, 3, 4, 0, 8, 1]