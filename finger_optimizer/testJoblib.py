# testJoblib.py
from joblib import Parallel, delayed, parallel_config
import numpy as np
from math import sqrt

def f(x):
    print(sqrt(x))

with parallel_config(backend='threading', njobs=5):
    Parallel()(delayed(f)(i**2) for i in range(10))
