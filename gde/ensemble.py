import gc
import logging
import os
import random
import socket
from datetime import datetime
from logging import config as init_logging_config
from pathlib import Path

import numpy as np
import pandas as pd
import solt
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gde.data.data_provider import DataProvider
from gde.eval.utils import init_model as init_model_eval
from gde.scheduler import LRScheduler
from gde.utils import init_obj_cls, init_obj, accuracy, predict_from_loader, init_ensemble_from_previous
from gde.utils import set_bn_eval, set_bn_train


class DeepEnsemble:
    def __init__(self, cfg):
        self.cfg = cfg

        self.epoch = None
        self.lr = None
        self.k = 0
        self.init_k = 0

        self.model, self.optimizer, self.criterion = None, None, None
        self.train_df, self.val_df, self.train_df_fnorm = None, None, None
        self.train_ds, self.val_ds, self.val_df_fnorm = None, None, None

        self.train_loader, self.val_loader = None, None
        self.prev_preds_train = None
        self.prev_preds_val = None

        self.original_cwd = Path().cwd().absolute()
        # Changing cwd to the autogenerated snapshot dir as defined in the config
        Path(cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.snapshot_dir) / 'config.yaml', 'w') as f:
            OmegaConf.save(cfg, f)
        os.chdir(cfg.snapshot_dir)

        self.cache_dir = None
        self.log_writer = None
        self.logger = None
        self.scheduler = None

        self.best_snapshot_fname = None
        self.best_val_metric = None

    def init_datasets(self):
        if self.cfg.data.data_dir is None:
            data_dir = self.original_cwd / 'datasets'
        else:
            data_dir = Path(self.cfg.data.data_dir)

        data_provider = DataProvider(self.cfg.data.dataset, data_dir, self.cfg.seed,
                                     self.cfg.data.val_amount,
                                     self.logger, self.cfg.data.std_mult_fnorm)

        return data_provider.init_splits()

    def init_loaders(self):
        if self.k == 0 or self.cfg.train.init_from is not None:
            self.train_df, self.val_df, self.train_df_fnorm, self.val_df_fnorm = self.init_datasets()

        if self.cfg.data.augs.train is not None:
            train_transforms = solt.utils.from_yaml(self.cfg.data.augs.train)
        else:
            train_transforms = solt.Stream()
        if self.cfg.data.augs.val is not None:
            val_transforms = solt.utils.from_yaml(self.cfg.data.augs.val)
        else:
            val_transforms = solt.Stream()

        dataset_cls = init_obj_cls(self.cfg.data.dataset_cls)

        if self.cfg.data.mean is not None:
            self.cfg.data.mean = tuple(self.cfg.data.mean)
            self.cfg.data.std = tuple(self.cfg.data.std)

        # This code assumes that validation transforms are NOT stochastic
        self.train_ds = dataset_cls(self.train_df, self.train_df_fnorm,
                                    train_transforms, val_transforms,
                                    self.prev_preds_train,
                                    mean=self.cfg.data.mean, std=self.cfg.data.std)

        self.val_ds = dataset_cls(self.val_df, self.val_df_fnorm,
                                  val_transforms, val_transforms,
                                  self.prev_preds_val,
                                  mean=self.cfg.data.mean, std=self.cfg.data.std)

        dataloader_cls = init_obj_cls(self.cfg.data.dataloader_cls)
        self.train_loader = dataloader_cls(
            self.train_ds, batch_size=self.cfg.data.batch_size, shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=False)

        self.val_loader = dataloader_cls(
            self.val_ds, batch_size=self.cfg.data.batch_size, shuffle=False,
            num_workers=self.cfg.data.num_workers, pin_memory=False)

        self.logger.info('Loaders have been initialized')

    def init_tensorboard(self):
        logdir = self.cache_dir / 'tb_logs' / f'run_{self.k}'
        logdir.mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(logdir)

    def cleanup(self):
        del self.model, self.optimizer, self.criterion
        gc.collect()
        torch.cuda.empty_cache()
        self.model, self.optimizer, self.criterion = None, None, None

    def init_model(self):
        # Model
        self.model = init_obj(self.cfg.model.cls, self.cfg.model.params)
        self.model.cuda()

    def init_kth_iteration(self):
        self.cleanup()
        # Resetting the seeds for randomization
        random.seed(self.cfg.seed + self.k)
        np.random.seed(self.cfg.seed + self.k)
        torch.cuda.manual_seed(self.cfg.seed + self.k)
        torch.manual_seed(self.cfg.seed + self.k)

        self.init_tensorboard()

        self.criterion = init_obj(self.cfg.criterion.cls, self.cfg.criterion.params)
        # Model is initialized in a separate method
        self.init_model()

        # Optimizer
        opt_params = dict(self.cfg.optimizer.params)
        opt_params['params'] = self.model.parameters()
        self.optimizer = init_obj(self.cfg.optimizer.cls, opt_params)

        self.lr = self.cfg.lr

        self.scheduler = LRScheduler(self)

        self.best_snapshot_fname.append(None)
        self.best_val_metric.append(None)

        if self.cfg.ensemble.greedy:
            if self.k > 0:
                self.logger.info(f'Loading predictions from iteration {self.k - 1}')
                p = pd.read_pickle(self.cache_dir / f'preds_{self.k - 1}_train.pkl')
                self.prev_preds_train.append(p)

                p = pd.read_pickle(self.cache_dir / f'preds_{self.k - 1}_val.pkl')
                self.prev_preds_val.append(p)

        self.init_loaders()

    def train(self):
        self.best_snapshot_fname = []
        self.best_val_metric = []
        self.prev_preds_train = []
        self.prev_preds_val = []

        # This ensures that the timing is the same in all the processes
        # Time is used to generate the snapshot names.
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        self.cache_dir = Path(f'deep_ensemble_cache_{dt_string}')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logging_conf = self.cfg.logging
        log_path = str(self.original_cwd / self.cfg.snapshot_dir / 'runs.log')
        logging_conf.handlers.file.filename = log_path
        logging_conf = OmegaConf.to_container(logging_conf, resolve=True)
        init_logging_config.dictConfig(logging_conf)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f'Host: {socket.gethostname()}')
        self.logger.info(f'Writing log to {log_path}')
        if 'SLURM_JOBID' in os.environ:
            self.logger.info(f'Running slurm job: {os.environ["SLURM_JOBID"]} | {os.environ["SLURM_PROCID"]}')

        # This part of the code looks at the previous snapshot and reads the config.
        # It copies all the snapshots from the previous iteration, so as the function norm predictions
        # The idea here is that we can train our pipeline on slurm clusters w/o any time limitations.
        if self.cfg.train.init_from is not None:
            init_from = self.original_cwd / Path(self.cfg.train.init_from)
            init_ensemble_from_previous(self, init_from)

        for self.k in range(self.init_k, self.cfg.ensemble.ens_size):
            self.init_kth_iteration()

            for self.epoch in range(self.cfg.train.num_epochs):
                self.scheduler.adjust_lr()

                self.model.train()
                train_loss, train_diversity = self.train_epoch()
                gc.collect()

                self.model.eval()
                val_loss, val_acc, val_diversity = self.val_epoch()
                gc.collect()

                # Saving the state
                self.save_state(val_acc)
                # Logging
                self.log_writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss},
                                            global_step=self.epoch)
                if val_diversity is not None:
                    self.log_writer.add_scalar(f'Auxiliary/diversity', val_diversity,
                                               global_step=self.epoch)
                self.log_writer.add_scalar(f'Metrics/accuracy', val_acc,
                                           global_step=self.epoch)
                # Logging the metrics
                log_out = f"[Model {self.k}][Epoch {self.epoch}] lr: {self.lr:.4f}"
                log_out += f"--train loss: {train_loss:.4f}"
                log_out += f"--val loss: {val_loss:.4f}"
                log_out += f"--val acc: {val_acc:.4f}"
                if val_diversity is not None:
                    log_out += f"--diversity: {val_diversity:.4f}"
                self.logger.info(log_out)

            if self.cfg.ensemble.greedy:
                self.collect_predictions()
            if self.cfg.train.break_every_iter:
                break

    def step_batch(self, batch, train=True):
        diversity = None
        data = batch['data'].cuda()
        targets = batch['target'].cuda()

        output = self.model(data)
        loss = self.criterion(output.squeeze(), targets)
        if train:
            loss.backward()

        if self.cfg.ensemble.greedy and self.k > 0:
            prev_preds_batch = {x: batch[x].cuda() for x in batch if x.startswith('pred_')}
            data_fnorm = batch['data_fnorm'].cuda()
            # Trick to avoid batchnorm getting crazy
            self.model.apply(set_bn_eval)
            output_fnorm = self.model(data_fnorm)
            self.model.apply(set_bn_train)
            diversity_term_preds = []
            for p_key in prev_preds_batch:
                d_term_j = output_fnorm.softmax(1) - prev_preds_batch[p_key].softmax(1)
                d_term_j = (d_term_j * d_term_j).sum(1).mean(0)
                diversity_term_preds.append(-self.cfg.ensemble.diversity_lambda / self.cfg.ensemble.k * d_term_j)
            if len(diversity_term_preds) > 0:
                diversity_term_preds = torch.stack(diversity_term_preds, dim=0)
                diversity = torch.logsumexp(diversity_term_preds, dim=0)

        if diversity is not None and train:
            diversity.backward()

        if train:
            self.optimizer.step()
            self.optimizer.zero_grad()
        if diversity is not None:
            loss += diversity
        return output, targets, loss, diversity

    def train_epoch(self):

        pbar = tqdm(total=len(self.train_loader))

        diversity = None
        running_loss = torch.tensor(0., requires_grad=False).cuda()
        running_diversity = torch.tensor(0., requires_grad=False).cuda()

        for i, batch in enumerate(self.train_loader):
            outputs, targets, loss, diversity = self.step_batch(batch)
            if diversity is not None:
                running_diversity += diversity.item()

            running_loss += loss.item()
            cur_loss = running_loss.item() / (i + 1)

            desc = f'[Model {self.k}][{self.epoch}] Train {loss.item():.4f} / {cur_loss:.4f}'
            pbar.set_description(desc)
            pbar.update()

        pbar.close()
        running_loss = running_loss / len(self.train_loader)
        running_diversity = running_diversity / len(self.train_loader)

        if diversity is not None:
            diversity = running_diversity.item()

        return running_loss.item(), diversity

    def val_epoch(self):
        pbar = tqdm(total=len(self.val_loader))

        diversity = None
        running_loss = 0
        running_acc = 0
        running_diversity = torch.tensor(0., requires_grad=False).cuda()

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                outputs, targets, loss, diversity = self.step_batch(batch, False)
                running_acc += accuracy(outputs, targets)[0]
                running_loss += loss
                if diversity is not None:
                    running_diversity += diversity.item()

                desc = f'[Model {self.k}][{self.epoch}] Val'
                pbar.set_description(desc)
                pbar.update()

        pbar.close()

        running_acc = running_acc / len(self.val_loader)
        running_loss = running_loss / len(self.val_loader)
        running_diversity = running_diversity / len(self.val_loader)

        if diversity is not None:
            diversity = running_diversity.item()

        return running_loss.item(), running_acc.item(), diversity

    def save_state(self, acc):
        model_name = self.cfg.model.name
        checkpoint_candidate_name = self.cache_dir / f'{self.k}_epoch_{self.epoch}_{model_name}_{acc:.4f}.pth'

        state = self.model.state_dict()

        if self.best_snapshot_fname[self.k] is None:
            self.logger.info(f'Saving the first snapshot [{checkpoint_candidate_name}]')
            torch.save(state, checkpoint_candidate_name)
            self.best_snapshot_fname[self.k] = checkpoint_candidate_name
            self.best_val_metric[self.k] = acc
        else:
            if acc > self.best_val_metric[self.k]:
                if not self.cfg.checkpointer.keep_old:
                    self.best_snapshot_fname[self.k].unlink()
                self.logger.info(f'Saving model [{checkpoint_candidate_name}]')
                self.best_snapshot_fname[self.k] = checkpoint_candidate_name
                self.best_val_metric[self.k] = acc
                torch.save(state, checkpoint_candidate_name)

    def collect_predictions(self):
        self.cleanup()
        model_kth = init_model_eval(self.cfg, self.best_snapshot_fname[self.k], 'cuda')

        # Sequential sampler is not needed, because we sort the returned items
        fnorm_train_loader = torch.utils.data.DataLoader(self.train_ds,
                                                         batch_size=self.cfg.data.batch_size,
                                                         num_workers=self.cfg.data.num_workers,
                                                         drop_last=False)

        fnorm_val_loader = torch.utils.data.DataLoader(self.val_ds,
                                                       batch_size=self.cfg.data.batch_size,
                                                       num_workers=self.cfg.data.num_workers,
                                                       drop_last=False)

        fnorm_preds_train = predict_from_loader(model_kth, fnorm_train_loader, False)
        fnorm_preds_val = predict_from_loader(model_kth, fnorm_val_loader, False)

        fnorm_preds_train.to_pickle(self.cache_dir / f'preds_{self.k}_train.pkl')
        fnorm_preds_val.to_pickle(self.cache_dir / f'preds_{self.k}_val.pkl')