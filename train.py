import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing

import utils.Constant as CONSTANT
from dataloader import Dataset
from evaluation import Evaluator

import warnings

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    from experiment import EarlyStop, train_model
    from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed

    # read configs
    config = Config(main_conf_path='./', model_conf_path='model_config')
    # apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = ' '.join(sys.argv[1:]).split(' ')
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip('-')
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)
    gpu = config.get_param('Experiment', 'gpu')
    gpu = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get_param('Experiment', 'model_name')
    bert_model_name = config.get_param('Model', 'bert_model_name')     

    # set seed
    seed = config.get_param('Experiment', 'seed')
    set_random_seed(seed)

    # logger
    if not os.path.exists('saves'):
        os.makedirs('saves')
    log_dir = make_log_dir(os.path.join('saves', model_name))
    logger = Logger(log_dir)
    config.save(log_dir)
    
    # dataset
    dataset_name = config.get_param('Dataset', 'data_name')
    dataset = Dataset(config, **config['Dataset'])

    # early stop
    early_stop = EarlyStop(**config['EarlyStop'])

    # evaluator()
    evaluator = Evaluator(early_stop.early_stop_measure, **config['Evaluator'])

    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    import model

    MODEL_CLASS = getattr(model, model_name)

    # build model
    model = MODEL_CLASS(dataset, config['Model'], device)
    model.logger = logger
    
    ################################# TRAIN & PREDICT
    # Train continually..
    checkpoint_path = config.get_param('Experiment', 'checkpoint_path')
    if checkpoint_path:
        logger.info(f"Model will be restored from {checkpoint_path}")
        checkpoint_log_dir = '/'.join(checkpoint_path.split('/')[:3])
        checkpoint_num_epoch = checkpoint_path.split('/')[3].split('_')[0]
        model.restore(checkpoint_log_dir, checkpoint_num_epoch)
    else:
        logger.info("There is no checkpoint to restore from.")

    # train
    try:
        valid_score, train_time = train_model(model, dataset, evaluator, early_stop, logger, config)
    except (KeyboardInterrupt, SystemExit):
        valid_score, train_time = dict(), 0
        logger.info("학습을 중단하셨습니다.")

    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)
    logger.info('\nTotal training time - %d:%d:%d(=%.1f sec)' % (h, m, s, train_time))

    # test
    model.eval()
    model.restore(logger.log_dir)
    test_score = evaluator.evaluate(model, dataset, 'test')

    # show result
    evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
    evaluation_table.add_row('Valid', valid_score)
    evaluation_table.add_row('Test', test_score)

    # evaluation_table.show()
    logger.info(evaluation_table.to_string())
        
    logger.info("Saved to %s" % (log_dir))
