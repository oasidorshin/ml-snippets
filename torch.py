import torch
import os, random
import numpy as np
from tqdm import tqdm


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

    # tf32
    #torch.set_float32_matmul_precision("high") # tf32
    #torch.set_float32_matmul_precision("highest") # pure fp32
    

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


# Sigmoid func
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# Boilerplate model class
class BaseModel(nn.Module):
    def __init__(self, device):
        super(BaseModel, self).__init__()

        self.model = timm.create_model("model_name", pretrained=False)
        self.model.fc = nn.Linear(1024, 10)

        self.device = device
        self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x

    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))

    def save_weights(self, weights_path):
        torch.save(self.model.state_dict(), weights_path)

    def train_epoch(self, train_loader, loss_fn, optimizer):
        self.model.train()

        for X, target in tqdm(train_loader):
            X, target = X.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(X)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def eval_epoch(self, valid_loader):
        self.model.eval()

        preds_arr = []
        with torch.no_grad():
            for X, target in tqdm(valid_loader):
                X, target = X.to(self.device), target.to(self.device)
                preds = sigmoid(self.model(X))
                preds_arr.append(preds.detach().cpu().numpy())

        return np.concatenate(preds_arr)


# Compile in 2.0
torch._dynamo.reset()
compiled_model = torch.compile(model, mode="max-autotune")


# AMP
preds_arr = []
with torch.no_grad(), torch.amp.autocast(device, enabled=True):
    for X, target in tqdm(valid_loader):
        X, target = X.to(self.device), target.to(self.device)
        preds = sigmoid(self.model(X))
        preds_arr.append(preds.detach().cpu().numpy())
