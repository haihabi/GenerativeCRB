import common
from experiments import constants
from tqdm import tqdm
from experiments.neural_network.training.single_network_optimization import SingleNetworkOptimization


def regression_training(dataset_loader, regression_network, optimizer: SingleNetworkOptimization, loss_function):
    def epoch_loop():
        ma = common.MetricAveraging()
        for x, y in tqdm(dataset_loader):
            optimizer.opt.zero_grad()
            x = x.to(constants.DEVICE)
            y = y.to(constants.DEVICE)
            y_hat = regression_network(x)
            loss = loss_function(y, y_hat)
            loss.backward()
            optimizer.opt.step()
            ma.update_metrics({'loss': loss})
        return ma.result

    for i in range(optimizer.n_epochs):
        print(f"Starting Epoch: {i + 1} of {optimizer.n_epochs}")
        results = epoch_loop()
        print(results)
