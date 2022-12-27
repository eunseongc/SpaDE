import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
import utils.Constant as CONSTANT

from collections import OrderedDict
from dataloader import Dataset
from evaluation import Evaluator
from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='saves/LASER/210830-213652', required=True, metavar='P')
parser.add_argument('--vector', type=str, default=None, required=False)
parser.add_argument('--test_set', type=str, default=None, required=False)
parser.add_argument('--num_iter', type=int, default=20, required=False)
parser.add_argument('--num_expand', type=int, default=None, required=False)
parser.add_argument('--doc_id', type=str, default=None, required=False)
parser.add_argument('--gpu', type=int, default=None, required=False) 
parser.add_argument('--expand_collection', type=str, default=None, required=False)
parser.add_argument('--alpha', type=float, required=False, default=None, help='Value for weighted average')
args = parser.parse_args()

# read configs
config = Config(main_conf_path=args.path, model_conf_path=args.path)
if args.test_set is not None:
    print(f'test_set will be changed from "{config.main_config["Dataset"]["test_set"]}" to "{args.test_set}"')
    config.main_config['Dataset']['test_set'] = args.test_set

if args.vector is not None:
    print(f'vector will be changed from "{config.main_config["Dataset"]["vector"]}" to "{args.vector}"')
    config.main_config['Dataset']['vector'] = args.vector

if args.doc_id is not None:
    print(f'doc_id will be changed from "{config.main_config["Dataset"]["doc_id"]}" to "{args.doc_id}"')
    config.main_config['Dataset']['doc_id'] = args.doc_id

if args.gpu is not None:
    print(f'gpu will be changed from "{config.main_config["Experiment"]["gpu"]}" to "{args.gpu}"')
    config.main_config['Experiment']['gpu'] = args.gpu

if args.num_expand is not None:
    print(f'num_expand will be changed from "{config.model_config["Model"]["num_expand"]}" to "{args.num_expand}"')
    config.model_config['Model']['num_expand'] = args.num_expand

if args.alpha is not None:
    print(f'alpha will be changed from "{config.model_config["Model"]["alpha"]}" to "{args.alpha}"')
    config.model_config['Model']['alpha'] = args.alpha

gpu = config.get_param('Experiment', 'gpu')
gpu = str(gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_name = config.get_param('Experiment', 'model_name') 
bert_model_name = config.get_param('Model', 'bert_model_name')

log_dir = args.path
logger = Logger(log_dir)

# dataset
vector = config.get_param('Dataset', 'vector')
data_name = config.get_param('Dataset', 'data_name')

if args.expand_collection is not None:
    dataset = Dataset(bert_model_name, **config['Dataset'], model_name=model_name, expand_collection=True)
else:    
    dataset = Dataset(bert_model_name, **config['Dataset'], model_name=model_name)

# evaluator
evaluator = Evaluator(**config['Evaluator'])

import model

MODEL_CLASS = getattr(model, model_name)

# build model
model = MODEL_CLASS(dataset, config['Model'], device)

# test
model.eval()
model.restore(logger.log_dir, args.num_iter)
model.logger = logger
if args.doc_id == 'valid':
    test_score = evaluator.evaluate(model, dataset, 'valid')
else:
    test_score = evaluator.evaluate(model, dataset, 'test')
# show result
evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
evaluation_table.add_row('Test', test_score)

# evaluation_table.show()
logger.info(f"> test set: {config.main_config['Dataset']['test_set']}")
logger.info(f"> vector: {config.main_config['Dataset']['vector']}")
logger.info(f"> num_iter: {args.num_iter}")
logger.info(f"> alpha: {args.alpha}")
logger.info(evaluation_table.to_string())

logger.info("Saved to %s" % (log_dir))