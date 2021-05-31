import os
import random
from pathlib import Path
from time import localtime, strftime

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

from gde.ensemble import DeepEnsemble

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

if __name__ == "__main__":
    OmegaConf.register_resolver("now", lambda pattern: strftime(pattern, localtime()))
    cwd = Path().cwd()
    p = cwd / "config" / "config.yaml"
    config = OmegaConf.load(str(p))
    conf_cli = OmegaConf.from_cli()
    for entry in config.defaults:
        assert len(entry) == 1
        for k, v in entry.items():
            if k in conf_cli:
                v = conf_cli[k]
            entry_path = cwd / "config" / k / f"{v}.yaml"
            entry_conf = OmegaConf.load(str(entry_path))
            config = OmegaConf.merge(config, entry_conf)

    cfg = OmegaConf.merge(config, conf_cli)
    snapshot_dir = Path(cfg.snapshot_dir)
    cfg.snapshot_dir = str(snapshot_dir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cudnn.benchmark = True

    ens = DeepEnsemble(cfg)
    ens.train()
