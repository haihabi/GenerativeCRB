import unittest

import torch

from data_model.linear_example import LinearFlow
import normalizing_flow as nf
import constants
import gcrb


class FlowToCRBTest(unittest.TestCase):
    def test_backward_flow_linear(self):
        lf = LinearFlow(2, 1, 0.1).to(constants.DEVICE)
        theta = 0.1 * torch.ones([1])
        gcrb_res = gcrb.compute_fim(lf, theta, 128)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
