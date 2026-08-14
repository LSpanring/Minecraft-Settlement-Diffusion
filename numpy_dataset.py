from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _identity(x):
    return x


class _TransformChain:
    """Picklable composition of two transforms — required for num_workers > 0 on Windows."""
    def __init__(self, base, extra):
        self.base = base
        self.extra = extra

    def __call__(self, x):
        return self.extra(self.base(x))


class NumpyPairDataset(Dataset):
    def __init__(
        self,
        path: str,
        targetDtype: Optional[np.dtype] = None,
        useMmap: bool = True,
        transform:  Callable[[torch.Tensor], torch.Tensor] = _identity,
        transform1: Callable[[torch.Tensor], torch.Tensor] = _identity,
        transform2: Callable[[torch.Tensor], torch.Tensor] = _identity,
    ):
        self._transform1 = _TransformChain(transform, transform1)
        self._transform2 = _TransformChain(transform, transform2)
        self._path = str(path)
        self._useMmap = useMmap

        self.data = np.load(path, mmap_mode=("r" if useMmap else None))

        self._targetDtype = targetDtype or self.data.dtype

    def __getstate__(self):
        # Don't pickle the mmap array — workers reopen the file from path.
        state = self.__dict__.copy()
        state['data'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = np.load(self._path, mmap_mode=("r" if self._useMmap else None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self._transform1(torch.from_numpy(self.data[index][0].astype(self._targetDtype, copy=True))),
            self._transform2(torch.from_numpy(self.data[index][1].astype(self._targetDtype, copy=True)))
        )
