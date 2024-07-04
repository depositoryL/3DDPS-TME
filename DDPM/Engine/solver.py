import os
import time
import torch
import numpy as np

from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from Utils.em_opt import traffic_em
from torch.nn.utils import clip_grad_norm_
from Utils.io import instantiate_from_config, get_model_parameters_info


def cycle(dl):
    while True:
        for data in dl:
            yield data

def expectation_maximization(
    flows_loads, 
    links_loads, 
    rm, 
    num_epoch: int
):
    b, device = flows_loads.shape[0], flows_loads.device
    flows_loads_final = torch.zeros_like(flows_loads, device=device)

    for i in range(b):
        flows_loads_i = em_iteration(flows_loads[i], links_loads[i], rm, num_epoch)
        flows_loads_final[i] = flows_loads_i

    return flows_loads_final

def em_iteration(x, y, rm, num_epoch):
    device, length = x.device, x.shape[0]
    idxes = torch.arange(0, length).to(device)
    rm = rm.to(device)
    rm, x_final = rm.to(device), x.clone()
    loss_min = torch.empty(length,).to(device)
    loss_min[:] = np.Inf

    for _ in range(num_epoch):
        a = x / rm.sum(dim=1)
        b = rm / (x @ rm).clamp_min_(1e-6).unsqueeze(1)

        c = y.unsqueeze(1) @ b.transpose(1, 2)
        x = torch.mul(a, c.squeeze(1))

        loss = torch.abs(x @ rm - y)
        loss = loss.sum(dim=1)

        select = (loss < loss_min).reshape(loss.shape)
        idx = idxes[select]

        if len(idx) != 0:
            loss_min[idx] = loss[idx]
            x_final[idx, :] = x[idx, :]

    return x_final



class Trainer(object):
    def __init__(self, model, config, args, data_loader, logger):
        super().__init__()
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.pre_epoch = config['solver']['max_epochs']
        self.train_num_steps = config['solver']['max_epochs']

        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = int(self.train_num_steps // 10)
        self.dl = cycle(data_loader)
        self.step = 0
        self.milestone = 0
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)
        
        self.criteon = nn.L1Loss().to(self.device)
        self.logger.log_info(str(get_model_parameters_info(self.model)))

    def save(self, milestone, loss):
        data = {
            'step': self.step,
            'loss': loss,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        self.milestone = milestone
        self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.model.load_state_dict(data['model'])

    def train(self):
        device = self.device        
        step = 0
        tic = time.time()

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    target = next(self.dl).to(device)
                    loss = self.model(target)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'Loss: {total_loss:.6f}')
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                step += 1
                self.step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone, loss)

                pbar.update(1)

        self.logger.log_info('Training complete, total time: {:.2f} s'.format(time.time() - tic))

    def sample(self, num, size_every, size):
        tic = time.time()
        self.logger.log_info('Begin to sample...')
        samples = np.empty([0, size[0], size[1]])
        num_cycle = int(num // size_every) + 1
        
        for _ in range(num_cycle):
            sample = self.ema.ema_model.sample(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        self.logger.log_info('Sampling done, time: {:.2f} s'.format(time.time() - tic))
        return samples
    
    def estimate(self, dataloader, A, learning_rate=1e-2, use_em=False):
        tic = time.time()
        self.logger.log_info('Begin to estimate...')
        model_kwargs = {}
        model_kwargs['learning_rate'] = learning_rate

        test_loss = []
        A = A.to(self.device)
        estimations = np.empty([0, dataloader.dataset.tm_dim])
        reals = np.empty([0, dataloader.dataset.tm_dim])
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x_hat = self.ema.ema_model.traffic_matrix_estimate(A, y, model_kwargs=model_kwargs)
            if use_em:
                x_hat = expectation_maximization(x_hat, y, A, 10)
            estimations = np.row_stack([estimations, x_hat.reshape(-1, x_hat.shape[-1]).detach().cpu().numpy()])
            reals = np.row_stack([reals, x.reshape(-1, x.shape[-1]).detach().cpu().numpy()])
            torch.cuda.empty_cache()
            test_loss_x = self.criteon(x_hat, x)
            test_loss.append(test_loss_x.item())

        test_loss = np.average(test_loss)
        self.logger.log_info('Estimation done, time: {:.2f} s, MAE: {:.4f}'.format(time.time() - tic, test_loss.item()))
        return estimations, reals

