import torch as t
import torch.nn as nn

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def tokenize(corpus: list[str]) -> list[list[str]]:
    tokenizer = get_tokenizer("basic_english")
    return list(map(tokenizer, corpus))


def build_vocab(tokens: list[list[str]], voc_size: int = 5000) -> torchtext.vocab.Vocab:
    vocabulary = build_vocab_from_iterator(tokens, max_tokens=voc_size, specials=["<unk>"])
    vocabulary.set_default_index(vocabulary["<unk>"])

    return vocabulary


def vocab_tokens(vocabulary: torchtext.vocab.Vocab, tokens: list[list[str]]) -> list[t.Tensor]:
    return [t.tensor(vocabulary(token), dtype=t.int64) for token in tokens]


def padding_indexes(tokens: list[t.Tensor], max_len: int) -> t.Tensor:
    embedding = []
    for token in tokens:
        # t.tensor(token, dtype=t.int64)
        embedding.append(nn.ConstantPad1d((max_len - len(token), 0), 0)(
            token.clone().detach().requires_grad_(False).to(t.int64))
        )
    return t.stack(embedding)
