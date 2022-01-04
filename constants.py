import common

DEVICE = common.get_working_device()
DIM = "dim"
THETA_MIN = "theta_min"
THETA_MAX = "theta_max"
THETA_DIM = "theta_dim"
SIGMA_N = "sigma_n"
K_SAMPLES = "k_samples"
M_SOURCES = "m_sources"
N_SENSORS = "n_sensors"
ARRAY_ARRANGEMENT = "array_arrangement"
LOGS = "logs"
DATASETS = "datasets"
PROJECT = "GenerativeCRB"
CROSS_POINT = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
ISO_LIST = [100, 400, 800, 1600, 3200]
CAM_DICT = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
INDEX2CAM = {v: k for k, v in CAM_DICT.items()}
ISO2INDEX = {100: 0,
             400: 1,
             800: 2,
             1600: 3,
             3200: 4}
