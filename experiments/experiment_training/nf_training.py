from experiments import constants
from normflowpy.nf_model import NormalizingFlowModel
from experiments.experiment_training.single_network_optimization import SingleNetworkOptimization
from experiments import common
from tqdm import tqdm
import copy


def normalizing_flow_training(flow_model: NormalizingFlowModel, training_dataset, validation_dataset,
                              flow_optimizer: SingleNetworkOptimization, check_gcrb=None):
    trm = common.TrainingResultsManger()
    best_model = copy.deepcopy(flow_model)

    def run_epoch():
        flow_model.train()
        print("Starting Training Loop")
        for x, theta in tqdm(training_dataset):
            flow_optimizer.zero_grad()
            x, theta = x.to(constants.DEVICE), theta.to(constants.DEVICE)
            loss = flow_model.nll_mean(x, **{constants.THETA: theta})
            loss.backward()
            grad_norm = flow_optimizer.step()

            trm.training_batch({"loss": loss, "grad_norm": grad_norm})
        flow_model.eval()
        print("Starting Validation Loop")
        for x, theta in tqdm(validation_dataset):
            x, theta = x.to(constants.DEVICE), theta.to(constants.DEVICE)
            val_nll = flow_model.nll_mean(x, **{constants.THETA: theta})
            trm.validation_batch({"loss": val_nll})
        flow_optimizer.end_epoch()

        return trm.end_epoch(additional_results_dict=check_gcrb(flow_model) if check_gcrb is not None else None)

    for i in range(flow_optimizer.n_epochs):
        print(f"Starting Epoch {i + 1} of {flow_optimizer.n_epochs}")
        is_best = run_epoch()
        if is_best:
            print("Update Best Model -:)")
            best_model = copy.deepcopy(flow_model)

    trm.print_best_values()
    return best_model, flow_model
