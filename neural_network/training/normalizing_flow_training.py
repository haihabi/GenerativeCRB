import common
import constants
from tqdm import tqdm
from neural_network.architecture import normalizing_flows as nf
from neural_network.training.single_network_optimization import SingleNetworkOptimization
from torch import nn


def flow_train(flow: nf.NormalizingFlow, dataset_loader, optimizer: SingleNetworkOptimization):
    def epoch_loop():
        ma = common.MetricAveraging()
        for x, y in tqdm(dataset_loader):
            optimizer.opt.zero_grad()
            x = x.to(constants.DEVICE)
            y = y.to(constants.DEVICE)

            nll = flow.nll(x, y).mean()

            nll.backward()
            nn.utils.clip_grad_norm_(
                flow.parameters(),
                0.1
            )
            optimizer.opt.step()
            ma.update_metrics({'nll': nll})
        return ma.result

    for i in range(optimizer.n_epochs):
        print(f"Starting Epoch: {i + 1} of {optimizer.n_epochs}")
        results = epoch_loop()
        print(results)
