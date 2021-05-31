import zipfile
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.datasets.utils import download_url

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class DataProvider(object):
    allowed_datasets = ["cifar10", "cifar100", "mnist", "two_moons"]

    def __init__(self, dataset, data_folder, seed, val_amount, logger, std_mult_fnorm=5):
        if dataset not in self.allowed_datasets:
            raise ValueError("Unsupported dataset")
        self.data_folder = data_folder
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.metadata = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.transforms = None
        self.val_amount = val_amount
        self.dataset = dataset
        self.log = logger
        self.std_mult_fnorm = std_mult_fnorm

    def _init_cifar(self, classes=10, data_folder=None, train=True):
        return self.get_pytorch_dataset_as_df(f"CIFAR{classes}", 32,
                                              train=train, data_folder=data_folder)

    def init_svhn(self, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder
        data_folder = Path(data_folder)
        if not (data_folder / 'svhn_cached.pkl').is_file():
            svhn_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
            download_url(svhn_url, str(data_folder), 'svhn_test.mat')
            loaded_mat = sio.loadmat(str(data_folder / 'svhn_test.mat'))
            data = np.transpose(loaded_mat['X'], (3, 0, 1, 2))
            labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(labels, labels == 10, 0)
            list_rows = [
                {"data": data[i, :, :, :].squeeze(), "target": labels[i]}
                for i in range(labels.shape[0])
            ]
            metadata = pd.DataFrame(list_rows)
            metadata.to_pickle(str(data_folder / 'svhn_cached.pkl'))
        else:
            metadata = pd.read_pickle(data_folder / 'svhn_cached.pkl')

        return metadata, 10

    def init_mnist(self, train=True, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder
        return self.get_pytorch_dataset_as_df("MNIST", 28, train=train, data_folder=data_folder)

    def init_fashion_mnist(self, train=True, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder
        return self.get_pytorch_dataset_as_df("FashionMNIST", 28, train=train, data_folder=data_folder)

    def init_omniglot(self, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder
        data_folder = Path(data_folder)
        if not (data_folder / 'omniglot_cached.pkl').is_file():
            omniglot_url = 'https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip'
            (data_folder / 'omniglot_images').mkdir(exist_ok=True)
            download_url(omniglot_url, str(data_folder), 'omniglot.zip')

            with zipfile.ZipFile(data_folder / 'omniglot.zip', 'r') as zip_ref:
                zip_ref.extractall(data_folder / 'omniglot_images')

            filelist = (data_folder / 'omniglot_images' / 'images_evaluation').glob('*/*/*.png')

            list_rows = [
                {"data": cv2.resize(255 - cv2.imread(str(fname)), (28, 28)), "target": 1} for fname in filelist
            ]
            metadata = pd.DataFrame(list_rows)
            metadata.to_pickle(str(data_folder / 'omniglot_cached.pkl'))
        else:
            metadata = pd.read_pickle(data_folder / 'omniglot_cached.pkl')

        return metadata, 0

    def init_lsun(self, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder
        data_folder = Path(data_folder)
        if not (data_folder / 'lsun_cached.pkl').is_file():
            lsun_url = 'http://dl.yf.io/lsun/scenes/test_lmdb.zip'
            download_url(lsun_url, str(data_folder), 'test_lmdb.zip')
            with zipfile.ZipFile(data_folder / 'test_lmdb.zip', 'r') as zip_ref:
                zip_ref.extractall(data_folder)
            (data_folder / 'test_lmdb.zip').unlink()
            env = lmdb.open(str(data_folder / 'test_lmdb'), max_readers=1, readonly=True, lock=False,
                            readahead=False, meminit=False)

            with env.begin(write=False) as txn:
                length = txn.stat()['entries']
            with env.begin(write=False) as txn:
                keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            list_rows = []
            for idx in range(length):
                with env.begin(write=False) as txn:
                    imgbuf = txn.get(keys[idx])
                image = np.frombuffer(imgbuf, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (32, 32))
                list_rows.append({'data': image, 'target': 1})
            metadata = pd.DataFrame(list_rows)
            metadata.to_pickle(str(data_folder / 'lsun_cached.pkl'))
        else:
            metadata = pd.read_pickle(data_folder / 'lsun_cached.pkl')

        return metadata, 0

    def get_pytorch_dataset_as_df(self, dataset_name, size, train=True, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder
        _db = getattr(datasets, dataset_name)(
            data_folder, train=train, transform=None, download=True
        )

        n_classes = np.unique(_db.targets).shape[0]
        if len(_db.data.shape) == 3:
            tmp = _db.data[:, :, :, None]
            _db.data = np.stack((tmp, tmp, tmp), axis=3).squeeze()
        if isinstance(_db.data, torch.Tensor):
            _db.data = _db.data.numpy()
        if isinstance(_db.targets, torch.Tensor):
            _db.targets = _db.targets.numpy()

        list_rows = [
            {"data": cv2.resize(_db.data[i, :, :, :].squeeze(), (size, size)), "target": _db.targets[i]}
            for i in range(len(_db.targets))
        ]
        metadata = pd.DataFrame(list_rows)

        return metadata, n_classes

    def init_cifar10(self, data_folder=None, train=True):
        return self._init_cifar(classes=10, data_folder=data_folder, train=train)

    def init_cifar100(self, data_folder=None, train=True):
        return self._init_cifar(classes=100, data_folder=data_folder, train=train)

    def init_cifar10_function_norm(self, data_folder=None):
        return self._init_image_dataframe_function_norm(data_folder)

    def init_cifar100_function_norm(self, data_folder=None):
        return self._init_image_dataframe_function_norm(data_folder)

    def init_mnist_function_norm(self, data_folder=None):
        return self._init_image_dataframe_function_norm(data_folder)

    def init_two_moons(self, data_folder=None):
        X, y = make_moons(300, random_state=self.seed, noise=0.3)

        return pd.DataFrame({'data': [np.array(x) for x in X.astype(np.float32).tolist()],
                             'target': y.astype(int).tolist()}), 2

    def init_two_moons_function_norm(self, data_folder=None):
        X = np.vstack(self.init_two_moons()[0].data.values.tolist())
        X_fnorm = np.random.randn(*X.shape) * self.std_mult_fnorm * X.std(0) + X.mean(0)
        return pd.DataFrame({'data': [np.array(x) for x in X_fnorm.astype(np.float32).tolist()]})

    def _init_image_dataframe_function_norm(self, data_folder=None):
        if data_folder is None:
            data_folder = self.data_folder

        dataframe_metadata, _ = getattr(self, f"init_{self.dataset}")(
            data_folder=data_folder
        )

        x_all = np.stack([entry.data for _, entry in dataframe_metadata.iterrows()])
        x_fnorm = np.random.randn(*x_all.shape) * self.std_mult_fnorm * x_all.std(0) + x_all.mean(0)
        return pd.DataFrame({'data': [np.array(x) for x in x_fnorm.astype(np.uint8).tolist()]})

    def init_splits(self):
        if not (self.data_folder / f'{self.dataset}_train_{self.seed}.pkl').is_file() \
                and not (self.data_folder / f'{self.dataset}_val_{self.seed}.pkl').is_file():
            self.log.info(f'Getting {self.dataset}')
            metadata, n_classes = getattr(self, f"init_{self.dataset}")(
                data_folder=self.data_folder
            )

            train_df, val_df = train_test_split(
                metadata,
                test_size=self.val_amount,
                shuffle=True,
                random_state=self.seed,
                stratify=metadata.target,
            )

            train_df.reset_index(inplace=True, drop=True)
            val_df.reset_index(inplace=True, drop=True)
            self.log.info(f"Saving cached train splits to {self.data_folder}")
            train_df.to_pickle(self.data_folder / f'{self.dataset}_train_{self.seed}.pkl')
            val_df.to_pickle(self.data_folder / f'{self.dataset}_val_{self.seed}.pkl')

        if not (self.data_folder / f'{self.dataset}_train_fnorm_{self.seed}.pkl').is_file():
            val_df = pd.read_pickle(self.data_folder / f'{self.dataset}_val_{self.seed}.pkl')
            fnorm_dataset = getattr(self, f"init_{self.dataset}_function_norm")(
                data_folder=self.data_folder
            )
            train_df_fnorm, val_df_fnorm = train_test_split(
                fnorm_dataset,
                test_size=val_df.shape[0],  # Validation sets need to match
                shuffle=True,
                random_state=self.seed,
            )

            train_df_fnorm.reset_index(inplace=True, drop=True)
            val_df_fnorm.reset_index(inplace=True, drop=True)

            self.log.info(f"Saving function norm splits.. to {self.data_folder}")
            train_df_fnorm.to_pickle(self.data_folder / f'{self.dataset}_train_fnorm_{self.seed}.pkl')
            val_df_fnorm.to_pickle(self.data_folder / f'{self.dataset}_val_fnorm_{self.seed}.pkl')

        train_df = pd.read_pickle(self.data_folder / f'{self.dataset}_train_{self.seed}.pkl')
        val_df = pd.read_pickle(self.data_folder / f'{self.dataset}_val_{self.seed}.pkl')
        train_df_fnorm = pd.read_pickle(self.data_folder / f'{self.dataset}_train_fnorm_{self.seed}.pkl')
        val_df_fnorm = pd.read_pickle(self.data_folder / f'{self.dataset}_val_fnorm_{self.seed}.pkl')

        self.log.info(f"The split has been loaded from disk by the all processes")

        return train_df, val_df, train_df_fnorm, val_df_fnorm
