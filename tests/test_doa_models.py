import unittest
import torch
from data_model.doa import DOAModel, SensorsArrangement
import os


class TestDOA(unittest.TestCase):
    def test_load_and_save(self):
        dm = DOAModel(SensorsArrangement.RANDOM, 4, 1, 2, 0.1, 0.1)
        dm.save_data_model("")

        dm2 = DOAModel(SensorsArrangement.RANDOM, 4, 1, 2, 0.1, 0.1)
        dm2.load_data_model("")
        n_diff = torch.sum((dm2.signal_generator.nominal_position - dm.signal_generator.nominal_position) != 0).item()
        os.remove(f"{dm2.model_name}_model.pt")
        self.assertTrue(n_diff == 0)

    def test_sample(self):
        dm = DOAModel(SensorsArrangement.RANDOM, 4, 1, 2, 0.1, 0.1)
        ds = dm.build_dataset(20)


if __name__ == '__main__':
    unittest.main()
