import math
import pickle

import numpy as np
import scipy.sparse as sp


def load_data_and_info(data_file):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict['train'], data_dict['valid']
