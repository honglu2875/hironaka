import time
from collections import defaultdict

import torch


class Timer:
    """
        (Special thank to Panpan Huang who wrote the initial codes.)
        A simple timer class that logs into an external dict.
    """

    def __init__(self, name: str, log_dict: dict, active=True, use_cuda=False):
        self.name = name
        self._active = active
        self._log_dict = log_dict
        self._use_cuda = use_cuda

    def __enter__(self):
        if self._active:
            if self._use_cuda:
                torch.cuda.current_stream().synchronize()
            self.start = time.perf_counter()
            if self.name not in self._log_dict:
                self._log_dict[self.name] = .0

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._active:
            if self._use_cuda:
                torch.cuda.current_stream().synchronize()
            if self.name in self._log_dict or isinstance(self._log_dict, defaultdict):
                self._log_dict[self.name] += (time.perf_counter() - self.start) * 1000  # ms
