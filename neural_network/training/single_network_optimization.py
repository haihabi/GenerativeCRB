import torch
from enum import Enum
import constants


class OptimizerType(Enum):
    SGD = 0
    Adam = 1


class SchedulerType(Enum):
    MultiStep = 0


class SingleNetworkOptimization(object):
    def __init__(self, network: torch.nn.Module, n_epochs: int,
                 lr=1e-4, weight_decay=1e-3, optimizer_type: OptimizerType = OptimizerType.SGD, grad_norm_clipping=10,
                 betas=(0.9, 0.999), enable_lr_scheduler=False, gamma: float = 0.1, scheduler_steps: list = []):
        self.n_epochs = n_epochs
        self.network = network
        self.optimizer_type = optimizer_type
        if self.optimizer_type == OptimizerType.SGD:
            self.opt = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.0, nesterov=False,
                                       weight_decay=weight_decay)
        elif self.optimizer_type == OptimizerType.Adam:
            self.opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            raise NotImplemented
        self.grad_norm_clipping = grad_norm_clipping
        self.enable_lr_scheduler = enable_lr_scheduler
        self.scheduler_list = []
        if self.enable_lr_scheduler:
            self.scheduler_list.append(
                torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=scheduler_steps, gamma=gamma))
        self.norm_type = 2

    def end_epoch(self):
        [s.step() for s in self.scheduler_list]

    def step(self):

        if self.grad_norm_clipping > 0:
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(),
                                                                 max_norm=self.grad_norm_clipping).item()
        else:
            grad_norm = torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach(), self.norm_type).to(constants.DEVICE) for p in
                     self.network.parameters() if p.grad is not None]),
                self.norm_type)
        self.opt.step()
        return grad_norm
