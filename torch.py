import torch
import os, random
import numpy as np


def setup_everything():
    # Seed everything
    seed=42

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Cudnn + determinism
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    
    # Setup visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    

"""
Deterministic dataloader
Usage example: 
seed_worker, g = deterministic_dataloader(42)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
    num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
"""
def deterministic_dataloader(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return seed_worker, g