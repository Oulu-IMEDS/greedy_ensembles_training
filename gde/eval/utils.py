import argparse
import pathlib

import numpy as np
import solt
import torch
from torch.utils.data import SequentialSampler

from gde.utils import init_obj_cls, init_obj


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=pathlib.Path, default='workdir')
    parser.add_argument('--arch', default='PreResNet164', choices=['VGG16BN', 'PreResNet8',
                                                                   'PreResNet164', 'WideResNet28x10'])
    parser.add_argument('--corruption', default='')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument('--ens_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[5, 10, 21, 42, 84])
    parser.add_argument('--diversity_lambda', type=float, default=1)
    parser.add_argument('--save_prefix', default='')

    return parser.parse_args()


def nll(labels, outputs):
    labels = labels.astype(int)
    idx = (np.arange(labels.shape[0]), labels)
    ps = outputs[idx]
    return -np.sum(np.log(ps))


def init_loader(cfg, df_to_predict, num_workers, batch_size):
    dataset_cls = init_obj_cls(cfg.data.dataset_cls)
    if cfg.data.augs.val is not None:
        val_transforms = solt.utils.from_yaml(cfg.data.augs.val)
    else:
        val_transforms = solt.Stream()
    ds_to_predict = dataset_cls(df_to_predict, df_to_predict, val_transforms, val_transforms, None,
                                mean=cfg.data.mean, std=cfg.data.std)
    sampler = SequentialSampler(ds_to_predict)
    loader = torch.utils.data.DataLoader(ds_to_predict, num_workers=num_workers,
                                         sampler=sampler, batch_size=batch_size)

    return loader


def predict_from_loader(model, loader):
    with torch.no_grad():
        preds = []
        gt = []
        for batch in loader:
            data, target = batch['data'], batch['target']
            preds.append(model(data.to('cuda')).to('cpu'))
            gt.append(target)
        preds = torch.cat(preds).softmax(1)
        gt = torch.cat(gt)
    return preds, gt


def init_model(cfg, snapshot, device='cuda'):
    model = init_obj(cfg.model.cls, cfg.model.params)
    state_dict = torch.load(snapshot, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
