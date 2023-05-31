import os
import argparse
import torch

from dataloader import Dataset
from evaluation import Evaluator
from utils import Config, Logger, ResultTable,


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, required=True, metavar='P')
parser.add_argument('--test_set', type=str, default=None, required=False)
parser.add_argument('--num_iter', type=int, default=None, required=True)
parser.add_argument('--doc_id', type=str, default='full', required=False)
parser.add_argument('--gpu', type=int, default=None, required=False) 
parser.add_argument('--expand_collection', type=str, default=None, required=False)
parser.add_argument('--alpha', type=float, required=False, default=None, help='Value for weighted average')
args = parser.parse_args()

# read configs
config = Config(main_conf_path=args.path, model_conf_path=args.path)
if args.test_set is not None:
    print(f'test_set will be changed from "{config.main_config["Dataset"]["test_set"]}" to "{args.test_set}"')
    config.main_config['Dataset']['test_set'] = args.test_set

if 'test' not in args.doc_id:
    print(f'doc_id will be changed from "{config.main_config["Dataset"]["doc_id"]}" to "{args.doc_id}"')
    config.main_config['Dataset']['doc_id'] = args.doc_id

if args.gpu is not None:
    print(f'gpu will be changed from "{config.main_config["Experiment"]["gpu"]}" to "{args.gpu}"')
    config.main_config['Experiment']['gpu'] = args.gpu

if args.alpha is not None:
    print(f'alpha will be changed from "{config.model_config["Model"]["alpha"]}" to "{args.alpha}"')
    config.model_config['Model']['alpha'] = args.alpha

gpu = config.get_param('Experiment', 'gpu')
gpu = str(gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = config.get_param('Experiment', 'model_name') 
bert_model_name = config.get_param('Model', 'bert_model_name')

log_dir = args.path
logger = Logger(log_dir)

# dataset
data_name = config.get_param('Dataset', 'data_name')
dataset = Dataset(config, **config['Dataset'])

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
test_score = evaluator.evaluate(model, dataset, mode='test')

# show result
evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
evaluation_table.add_row('Test', test_score)

# evaluation_table.show()
logger.info(f"> test set: {config.main_config['Dataset']['test_set']}")
logger.info(f"> num_iter: {args.num_iter}")
logger.info(f"> alpha: {args.alpha}")
logger.info(evaluation_table.to_string())

logger.info("Saved to %s" % (log_dir))