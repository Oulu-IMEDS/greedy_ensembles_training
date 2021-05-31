import gc
import logging
import pathlib
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf

from gde.data.data_provider import DataProvider
from gde.eval.utils import init_loader, predict_from_loader, init_model, init_args

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    args = init_args()
    assert args.corruption == ''
    results = []
    datasets_dict = {'cifar10': {5: None, 10: None, 21: None, 42: None, 84: None},
                     'cifar100': {5: None, 10: None, 21: None, 42: None, 84: None},
                     'mnist': {5: None, 10: None, 21: None, 42: None, 84: None}}

    arch = args.arch
    dataset = args.dataset
    ens_size = args.ens_size
    seed = args.seed

    if pathlib.Path(f'results-{args.arch}-{args.dataset}-{args.ens_size}-{seed}.pkl').is_file():
        logger.info('Results have already been computed. Exiting...')
        sys.exit(0)

    counter = 0
    exps = list(args.workdir.glob(f'*_{dataset}_{ens_size}_{arch}_*{seed}_{ens_size}/*/*'))
    total = len(exps)
    data_dir = pathlib.Path('datasets')

    logger.info(f'Architecture: {arch}')
    logger.info(f'Dataset: {dataset}')
    logger.info(f'Ens size: {dataset}')
    logger.info(f'Seed: {seed}')
    logger.info(f'Total of {total} experiments to run')
    logger.info(f'Data dir: {data_dir.absolute()}')

    if dataset == 'cifar10' or dataset == 'cifar100':
        ds1_label = 'svhn'
        ds2_label = 'lsun'
    elif dataset == 'mnist':
        ds1_label = 'fashion_mnist'
        ds2_label = 'omniglot'
    else:
        raise ValueError('Dataset is NOT supported')

    for exp in exps:
        snapshots = list(exp.glob('deep_ensemble_cache*/*.pth'))
        if len(snapshots) != ens_size:
            print(exp)
            continue
        cfg = OmegaConf.load(exp / 'config.yaml')
        method = 'greedy' if cfg.ensemble.greedy else 'deepens'
        logger.info(f'Doing experiment {counter + 1} / '
                    f'{total} [{arch}-{dataset}-{ens_size}-{method}-{seed}]')

        if datasets_dict[dataset][seed] is None:
            provider = DataProvider(cfg.data.dataset, data_dir, cfg.seed, cfg.data.augs,
                                    cfg.data.val_amount)
            # For testing
            test_df, _ = getattr(provider, f"init_{dataset}")(train=False)

            # For ood detection (two datasets always)
            # In the case of CIFAR, df1 is svhn and df2 is lsun

            ood_df1, _ = getattr(provider, f'init_{ds1_label}')(data_dir)
            ood_df2, _ = getattr(provider, f'init_{ds2_label}')(data_dir)

            ood_df1['target'] = 1
            id_df = test_df.copy()
            id_df['target'] = 0
            ood_df1 = pd.concat((ood_df1, id_df), axis=0)

            ood_df2['target'] = 1
            id_df = test_df.copy()
            id_df['target'] = 0
            ood_df2 = pd.concat((ood_df2, id_df), axis=0)

            test_loader = init_loader(cfg, test_df, 4, 128)
            ood_loader1 = init_loader(cfg, ood_df1, 4, 128)
            ood_loader2 = init_loader(cfg, ood_df2, 4, 128)

            datasets_dict[dataset][seed] = (test_loader, ood_loader1, ood_loader2)
        else:
            test_loader, ood_loader1, ood_loader2 = datasets_dict[dataset][seed]

        with torch.no_grad():
            preds_ens = []
            preds_ood_ens_ds1 = []
            preds_ood_ens_ds2 = []
            for model_snapshot in snapshots:
                model = init_model(cfg, model_snapshot)
                preds, gt = predict_from_loader(model, test_loader)

                preds_ood_ds1, gt_ood_ds1 = predict_from_loader(model, ood_loader1)
                preds_ood_ds2, gt_ood_ds2 = predict_from_loader(model, ood_loader2)

                preds_ood_ens_ds1.append(preds_ood_ds1)
                preds_ood_ens_ds2.append(preds_ood_ds2)
                preds_ens.append(preds)

            preds_ens = torch.stack(preds_ens, 0).numpy()
            preds_ood_ens_ds1 = torch.stack(preds_ood_ens_ds1, 0).numpy()
            preds_ood_ens_ds2 = torch.stack(preds_ood_ens_ds2, 0).numpy()

            gt = gt.numpy()

        if cfg.ensemble.greedy:
            div_lambda = cfg.ensemble.diversity_lambda
        else:
            div_lambda = 'N/A'

        prior_lambda = cfg.train.prior_lambda
        results.append([arch, dataset, seed, prior_lambda, ens_size, method, div_lambda,
                        preds_ens, preds_ood_ens_ds2, preds_ood_ens_ds1,
                        gt, gt_ood_ds2, gt_ood_ds1])
        gc.collect()
        torch.cuda.empty_cache()
        counter += 1

    results = pd.DataFrame(
        columns=['Architecture', 'Dataset', 'Seed', 'Prior',
                 'Size', 'Method', 'Diversity',
                 'Preds_ens', f'Preds_ood_{ds2_label}', f'Preds_ood_{ds1_label}',
                 'Gt_ens', f'Gt_ood_{ds2_label}', f'Gt_ood_{ds1_label}'],
        data=results)

    results.to_pickle(f'results-{args.arch}-{args.dataset}-{args.ens_size}-{seed}.pkl')
