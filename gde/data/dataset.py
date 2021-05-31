import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import solt
import torch.utils
from omegaconf import ListConfig
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class DataFrameDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame,
                 metadata_fnorm: Optional[pd.DataFrame] = None,
                 transforms: Optional[solt.Stream] = None,
                 fnorm_transforms: Optional[solt.Stream] = None,
                 metadata_prev: Optional[List[pd.DataFrame]] = None,
                 data_key: str = 'data',
                 target_key: Optional[str] = 'target',
                 mean=None, std=None):

        self.metadata = metadata
        self.metadata_fnorm = metadata_fnorm
        self.metadata_prev = metadata_prev
        self.data_key = data_key
        self.target_key = target_key
        self.transforms = transforms
        self.fnorm_transforms = fnorm_transforms

        if isinstance(mean, ListConfig):
            mean = list(mean)
            std = list(std)

        if mean is not None:
            if not isinstance(mean, (list, tuple)):
                mean = [mean, ]
            if not isinstance(std, (list, tuple)):
                std = [std, ]

        if isinstance(mean, list):
            mean = tuple(mean)
            std = tuple(std)

        self.mean = mean
        self.std = std

    def read_data(self, entry, fnorm=False):
        return getattr(entry, self.data_key).astype(np.float32)

    def read_pred(self, entry):
        # Predicts are stored either as floats for regression
        # or as numpy tensors for classification
        if isinstance(entry.pred, np.ndarray):
            return entry.pred.astype(np.float32)
        return float(entry.pred)

    def __getitem__(self, idx):
        entry = self.metadata.iloc[idx]
        res = {'data': self.read_data(entry), 'idx': idx}
        if self.metadata_fnorm is not None:
            entry_fnorm = self.metadata_fnorm.iloc[idx % self.metadata_fnorm.shape[0]]
            res['idx_fnorm'] = idx % self.metadata_fnorm.shape[0]
            res['data_fnorm'] = self.read_data(entry_fnorm, True)

        if self.target_key is not None:
            res['target'] = getattr(entry, self.target_key)
        # Getting the predictions from the previous models
        if self.metadata_prev is not None:
            for prev_model_idx, prev_meta in enumerate(self.metadata_prev):
                if self.metadata_fnorm is None:
                    res[f'pred_{prev_model_idx}'] = self.read_pred(prev_meta.iloc[idx])
                else:
                    res[f'pred_{prev_model_idx}'] = self.read_pred(prev_meta.iloc[idx % self.metadata_fnorm.shape[0]])

        return res

    def __len__(self):
        return self.metadata.shape[0]


class DataFrameImageDataset(DataFrameDataset):
    def __init__(self, metadata: pd.DataFrame,
                 metadata_fnorm: Optional[pd.DataFrame],
                 transforms: solt.Stream,
                 fnorm_transforms: Optional[solt.Stream] = None,
                 metadata_prev: Optional[List[pd.DataFrame]] = None,
                 data_key: str = 'data',
                 target_key: Optional[str] = 'target',
                 mean=None, std=None):
        super(DataFrameImageDataset, self).__init__(metadata,
                                                    metadata_fnorm,
                                                    transforms,
                                                    fnorm_transforms,
                                                    metadata_prev,
                                                    data_key,
                                                    target_key,
                                                    mean,
                                                    std)

    def read_data(self, entry, fnorm=False):
        img = getattr(entry, self.data_key)
        if fnorm:
            return self.fnorm_transforms(img, mean=self.mean, std=self.std)['image']
        return self.transforms(img, mean=self.mean, std=self.std)['image']

    def read_pred(self, entry):
        # Predicts are stored either as floats for regression
        # or as numpy tensors for classification
        assert isinstance(entry.pred, np.ndarray)
        assert entry.pred.dtype == np.float32
        return entry.pred


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
