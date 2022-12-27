import copy
from time import time

# https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb#scrollTo=6Pi9ebpTB9em

def train_model(model, dataset, evaluator, early_stop, logger, config):
    # train model with the given experimental setting
    # input: sess, initialized model, dataset, evaluator, early_stop, logger, config
    # output: optimized model, best_valid_score
    logger.info('train start ... !')
    early_stop.initialize()
    valid_score, train_time = model.train_model(dataset, evaluator, early_stop, logger, config)

    return valid_score, train_time