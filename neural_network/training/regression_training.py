import common
import constants
from tqdm import tqdm
from neural_network.training.single_network_optimization import SingleNetworkOptimization


def regression_training(dataset_loader, regression_network, optimizer: SingleNetworkOptimization, loss_function,
                        n_epochs):
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

    for i in range(n_epochs):
        print(f"Starting Epoch: {i + 1} of {n_epochs}")
        results = epoch_loop()
        print(results)
