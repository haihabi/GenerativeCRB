import matplotlib.pyplot as plt

from common.metric_averaging import MetricAveraging
import wandb

TRAINING = "training_"
VALIDATION = "validation_"


class TrainingResultsManger(object):
    def __init__(self):
        self.training_ma = MetricAveraging()
        self.val_ma = MetricAveraging()
        self.results_dict = dict()
        self.best_value_training = None
        self.best_value_val = None

    def print_best_values(self):
        print("-" * 100)
        print(f"Best Loss Validation:{self.best_value_val} and Training {self.best_value_training}")
        print("-" * 100)

    def training_batch(self, results_dict):
        self.training_ma.update_metrics(results_dict)

    def validation_batch(self, results_dict):
        self.val_ma.update_metrics(results_dict)

    def update_best_val(self, v):
        is_best = False
        if self.best_value_val is None:
            self.best_value_val = v
            is_best = True
        elif v < self.best_value_val:
            self.best_value_val = v
            is_best = True
        return is_best

    def update_best_training(self, v):
        is_best = False
        if self.best_value_training is None:
            self.best_value_training = v
            is_best = True
        elif v < self.best_value_training:
            self.best_value_training = v
            is_best = True
        return is_best

    def end_epoch(self, results2point=("loss",), best_metric="loss", additional_results_dict=None):
        training_results_dict = self.training_ma.result
        validation_results_dict = self.val_ma.result
        print("-" * 100)
        is_best_val = False
        is_best_training = False
        results_dict2log = {}
        for k, v in training_results_dict.items():
            self._append2results_dict(TRAINING + k, v)
            results_dict2log.update({TRAINING + k: v})
            if k in results2point:
                print(f"Training-{k}:{v}")
            if k == best_metric:
                is_best_training = self.update_best_training(v)
        for k, v in validation_results_dict.items():
            self._append2results_dict(VALIDATION + k, v)
            results_dict2log.update({VALIDATION + k: v})
            if k in results2point:
                print(f"Validation-{k}:{v}")
            if k == best_metric:
                is_best_val = self.update_best_val(v)
        is_best = is_best_val and is_best_training
        if additional_results_dict is not None:
            results_dict2log.update(additional_results_dict)
        wandb.log(results_dict2log)
        return is_best

    def _append2results_dict(self, k, v):
        if self.results_dict.get(k) is None:
            self.results_dict.update({k: []})
        self.results_dict[k].append(v)

    def plot(self, results_name):
        if len(results_name) == 1:
            raise NotImplemented
        else:
            for i, names in enumerate(results_name):
                plt.subplot(1, len(results_name), i + 1)
                for n in names:
                    plt.plot(self.results_dict[n], label=n)
                plt.grid()
                plt.legend()
                plt.xlabel("Epoch")
            plt.show()
