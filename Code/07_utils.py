import numpy as np
import random
import torch

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_60_20_20(X, y):
    n = len(X)
    a = int(n * 0.6)
    b = int(n * 0.8)
    return (
        X.iloc[:a], y.iloc[:a],
        X.iloc[a:b], y.iloc[a:b],
        X.iloc[b:], y.iloc[b:]
    )
