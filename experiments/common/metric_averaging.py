import torch


class SingleMetricAveraging(object):
    def __init__(self):
        self.n = 0
        self.accumulator = 0

    def update_metric(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.accumulator += value
        self.n += n

    @property
    def result(self):
        return self.accumulator / self.n


class MetricAveraging(object):
    def __init__(self):
        self.metric_dict = dict()

    def _get_current_metric(self, name) -> SingleMetricAveraging:
        if self.metric_dict.get(name) is None:
            self.metric_dict.update({name: SingleMetricAveraging()})
        return self.metric_dict[name]

    def update_metrics(self, results_dict):
        for k, v in results_dict.items():
            smv = self._get_current_metric(k)
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    smv.update_metric(v[0], v[1])
                elif len(v) == 1:
                    smv.update_metric(v[0])
                else:
                    raise Exception('Metric Dict is illegal')
            else:
                smv.update_metric(v)

    @property
    def result(self) -> dict:
        return {k: v.result for k, v in self.metric_dict.items()}
