import torch as t


def optimal_device():
    if t.cuda.is_available():
        return t.device('cuda')
    else:
        try:
            return t.device('mps')
        except Exception as _:
            return t.device('cpu')
