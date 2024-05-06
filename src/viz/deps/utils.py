import time
from typing import Callable
from functools import wraps


def benchmark(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        try:
            v = func(*args, **kwargs)
            return v
        finally:
            print(f'Time taken for {func.__name__}: {time.time() - t}')
    return wrapper
