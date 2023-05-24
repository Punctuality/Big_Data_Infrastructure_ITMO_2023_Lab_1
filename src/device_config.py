import torch as t

import logging as log

def optimal_device():
    log.debug("Checking for CUDA availability")
    if t.cuda.is_available():
        log.debug("CUDA is available")
        return t.device('cuda')
    else:
        log.debug("CUDA is NOT available")
        try:
            dev = t.device('mps')
            log.debug("Fallbacking to to use MPS")
            return dev
        except Exception as _:
            log.debug("Fallbacking to CPU")
            return t.device('cpu')
