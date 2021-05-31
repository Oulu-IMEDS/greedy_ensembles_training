import importlib
import shutil

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_obj_cls(string_def):
    string_parts = string_def.split(".")
    obj_cls = getattr(importlib.import_module(".".join(string_parts[:-1])), string_parts[-1])
    return obj_cls


def init_obj(string_def, params):
    obj_cls = init_obj_cls(string_def)
    if params is None:
        params = {}
    return obj_cls(**params)


def predict_from_loader(model, loader, fnorm_softmax=True):
    with torch.no_grad():
        preds = []
        indices = []
        device = next(model.parameters()).device
        for batch in tqdm(loader, desc='Computing preds from loader'):
            data = batch['data_fnorm'].to(device)
            idx = batch['idx']
            pred = model(data)
            if fnorm_softmax:
                pred = pred.softmax(1)
            preds.append(pred.to('cpu'))
            indices.append(idx)
        preds = torch.cat(preds, dim=0).squeeze()
        indices = torch.cat(indices, dim=0).squeeze()
        original_order = torch.argsort(indices)
        preds = [np.array(p).astype(np.float32) for p in preds[original_order].numpy().tolist()]
    return pd.DataFrame(data={'pred': preds})


def init_ensemble_from_previous(pipeline, init_from):
    if pipeline.global_rank == 0:
        pipeline.logger.info(f'Found a breakpoint ot start from: loading from {init_from}')
    # Reading the config first
    cfg_prev_path = list(init_from.glob('*/*/config.yaml'))
    assert len(cfg_prev_path) == 1
    cfg_prev_path = cfg_prev_path[0]
    if pipeline.global_rank == 0:
        pipeline.logger.info(f'Found config {cfg_prev_path}')
    cfg_prev = OmegaConf.load(str(cfg_prev_path))
    assert cfg_prev.ensemble.k == pipeline.cfg.ensemble.k

    predictions_prev = list(init_from.glob('*/*/*/*.pkl'))
    snapshots_prev = list(init_from.glob('*/*/*/*.pth'))
    pipeline.init_k = len(snapshots_prev)
    pipeline.k = pipeline.init_k

    if pipeline.global_rank == 0:
        pipeline.logger.info(f'Will start ensemble iteration from from k = {pipeline.init_k}')
        for pred_path in predictions_prev:
            pipeline.logger.info(f'Copying {pred_path} to {pipeline.cache_dir / pred_path.name}')
            shutil.copy(pred_path, pipeline.cache_dir / pred_path.name)

        for snp_path in snapshots_prev:
            pipeline.logger.info(f'Copying {snp_path} to {pipeline.cache_dir / snp_path.name}')
            shutil.copy(snp_path, pipeline.cache_dir / snp_path.name)

            pipeline.best_snapshot_fname.append(None)
            pipeline.best_val_metric.append(None)
    # This line ensures that everything has been copied from the previous snapshot
    pipeline.barrier()
    if pipeline.cfg.ensemble.greedy:
        for prev_k in range(0, pipeline.init_k):
            p = pd.read_pickle(pipeline.cache_dir / f'preds_{prev_k}_train.pkl')
            pipeline.prev_preds_train.append(p)

            p = pd.read_pickle(pipeline.cache_dir / f'preds_{prev_k}_val.pkl')
            pipeline.prev_preds_val.append(p)

    pipeline.barrier()


def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()
