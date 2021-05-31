import numpy as np


class LRScheduler:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def lr_schedule_epoch(self):
        epoch = self.pipeline.epoch
        cfg = self.pipeline.cfg

        lr = cfg.optimizer.params.lr
        if cfg.optimizer.scheduler.type == 'milestones':
            scheduler_conf = cfg.optimizer.scheduler
            for epoch_drop in scheduler_conf.milestones:
                if epoch_drop - 1 <= epoch:
                    lr *= scheduler_conf.gamma
        elif cfg.optimizer.scheduler.type == 'annealing':
            # replication of the schedule from SWA paper
            t = epoch / cfg.train.num_epochs
            lr_scaler = cfg.optimizer.scheduler.lr_scaler
            t1 = cfg.optimizer.scheduler.t1
            t2 = cfg.optimizer.scheduler.t2

            # We run a constant high lr for 50% of the time
            # For the remaining 40%, we will linearly decrease it
            # The last 10% of the training time, we will just run training with low LR
            if t <= t1:
                factor = 1.0
            elif t <= t2:
                factor = 1.0 - (1.0 - lr_scaler) * (t - t1) / (t2 - t1)
            else:
                factor = lr_scaler

            lr = cfg.optimizer.params.lr * factor
        else:
            raise NotImplementedError('Unknown scheduler')

        warmup_for = cfg.optimizer.scheduler.warmup_for
        warmup_from = cfg.optimizer.scheduler.warmup_from
        if warmup_from is None:
            start_lr = cfg.optimizer.params.lr * 0.1
        else:
            start_lr = cfg.optimizer.scheduler.warmup_from

        if warmup_for != 0 and epoch < warmup_for:
            return start_lr + cfg.optimizer.params.lr * epoch / warmup_for

        return lr

    def adjust_lr(self):
        new_lr = self.lr_schedule_epoch()

        if new_lr != self.pipeline.lr:
            for param_group in self.pipeline.optimizer.param_groups:
                param_group["lr"] = new_lr

        # This assumes that we have only one lr for all the parameter groups!
        lrs = [pg["lr"] for pg in self.pipeline.optimizer.param_groups]
        lrs = np.unique(lrs)
        assert lrs.shape[0] == 1
        self.pipeline.lr = lrs[0]
