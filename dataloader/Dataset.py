import os
import gzip
import json
import pickle
from pyexpat import model

import numpy as np

from base import BaseDataset
from tqdm import tqdm
from time import time
from collections import defaultdict

from transformers import BertTokenizerFast
from scipy.sparse import csr_matrix
from utils.Tool import *

class Dataset(BaseDataset):
    def __init__(self, config, data_dir, data_name, triple_name, doc_id, test_set, expand_collection):
        super(Dataset, self).__init__(data_dir, data_name, triple_name, doc_id, test_set, expand_collection)
    
        self.config = config
        self.model_name = self.config.main_config['Experiment']['model_name']

        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model_config['Model']['bert_model_name'])
        self.word2idx = self.tokenizer.get_vocab()

        self.vocab_size = len(self.word2idx)

        if data_name == 'marco-passage':
            self.load_data_msmarco()
        else:
            raise ValueError(f"Invalid data_name: {data_name}")

    def load_data_msmarco(self):      
        # Loading Queries
        self.train_query_file = open(os.path.join(self.data_dir, self.data_name, 'queries.train.tsv'), encoding='utf8')
        self.query2text = {}
        for line in tqdm(self.train_query_file, desc='Loading queries..'):
            qid, qtext = line.rstrip().split('\t')
            self.query2text[qid] = qtext

        # Loading collections
        if self.model_name == "SpaDE_cotraining":
            collection_path=os.path.join(self.data_dir, self.data_name, 'collection.tsv')
            self.passage2text = {}
            for line in tqdm(open(collection_path), desc=f'Loading collection from {collection_path}...'):
                pid, passage = line.split('\t')
                self.passage2text[pid] = passage.rstrip()        

            collection_path=os.path.join(self.data_dir, self.data_name, 'expanded_collection.tsv')
            self.passage2text_exp = {}
            for line in tqdm(open(collection_path), desc=f'Loading collection from {collection_path}...'):
                pid, passage = line.split('\t')
                self.passage2text_exp[pid] = passage.rstrip()    
        else:
            collection_path=os.path.join(self.data_dir, self.data_name, 'collection.tsv')
            if self.expand_collection: collection_path=os.path.join(self.data_dir, self.data_name, 'expanded_collection.tsv')
                
            self.passage2text = {}
            for line in tqdm(open(collection_path), desc=f'Loading collection from {collection_path}...'):
                pid, passage = line.split('\t')
                self.passage2text[pid] = passage.rstrip()
            
        ## Loading triplesles.pkl
        triples_path = os.path.join(self.data_dir, self.data_name, f'{self.triple_name}_triples.pkl')

        print(f'Loading triples from: {triples_path}')
        if os.path.exists(triples_path):
            with open(triples_path, 'rb') as f:
                self.train_q_indices, self.train_pos_indices, self.train_neg_indices, \
                self.valid_q_indices, self.valid_pos_indices, self.valid_neg_indices = pickle.load(f)
                self.train_targets, self.valid_targets = None, None
        else:
            raise ValueError(f"Invalid triples_path: {triples_path}")
        
        ## Loading Doc IDs to create a document sparse matrix
        doc_id_path = f'doc_id_{self.doc_id}_sorted.pkl'

        with open(os.path.join(self.data_dir, self.data_name, doc_id_path), 'rb') as f:
            print(f'Loading {self.doc_id} doc_ids from {doc_id_path}')
            self.doc_id = pickle.load(f)

        # Loading queries for evaluation
        if self.test_set == 'dev':
            self.test_matrix, self.test_id = self.load_feat(os.path.join(self.data_dir, self.data_name, f"test_data/queries_dev.featpkl"))
            self.qid2pid = self.load_rels(os.path.join(self.data_dir, self.data_name, "test_data/rels_dev.txt"))
        else:
            print(f'Please set test set adequately')
            exit(1)

        query_feat_path = f'queries.train.featpkl'
        self.queries, self.query_ids = self.load_feat(os.path.join(self.data_dir, self.data_name, query_feat_path))
        
        self.train_qrels = dict()
        qrel_path = os.path.join(self.data_dir, self.data_name, 'qrels.train.tsv')
        ## pid = integer, qid = string
        self.qid2pids_train = defaultdict(list)
        with open(qrel_path) as f:
            for line in tqdm(f, desc="Loading qrels ..."):
                qid, _, pid, rel = line.strip().split('\t')
                if rel != '1':
                    continue
                self.qid2pids_train[qid].append(pid)

        doc_id_valid_path = os.path.join(self.data_dir, self.data_name, 'doc_id_valid.pkl')
        with open(doc_id_valid_path, 'rb') as f:
            print(f'Loading valid doc_ids!!')
            self.doc_id_valid = pickle.load(f)

        self.valid_newqids = np.unique(self.valid_q_indices)
        self.valid_matrix, self.valid_id = self.queries[self.valid_newqids], self.query_ids[self.valid_newqids]
        
        self.qid2pid_valid = dict()
        for qid, pids in tqdm(self.qid2pids_train.items(), dynamic_ncols=True, desc="> Processing qid2pid_valid"):
            if str(qid) in self.valid_id:
                self.qid2pid_valid[qid] = pids

    def __str__(self):
        ret_str = '\n'
        ret_str += 'Dataset: %s\n' % self.data_name
        # ret_str += '# of docs_data: %d\n' % len(self.docs_data)
        # ret_str += '# of rels_data: %d(%d+%d)\n' % (self.num_rels_train + self.num_rels_test, self.num_rels_train, self.num_rels_test)
        return ret_str


    def load_feat(self, data_path):
        print("Loading ", data_path)
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                return pickle.load(f)

        row, col, data, qids = [], [], [], []

        raw_data_path = '.'.join(data_path.split('.')[:-1])+'.tsv'
        for r, lin in enumerate(tqdm(open(raw_data_path), desc="> Creating query sparse matrix")):
            qid, content = lin.rstrip().split('\t')
            qids.append(qid)
            content = list(set(self.tokenizer.tokenize(content)))
            for word in content:
                if self.word2idx.get(word):
                    row.append(r)
                    col.append(self.word2idx[word])
                    data.append(1)

        data_mat = csr_matrix((data, (row, col)), shape=(max(row)+1, self.vocab_size))
        with open(data_path, 'wb') as f:
            pickle.dump([data_mat, np.array(qids)], f, protocol=4)
        return data_mat, np.array(qids)

    def load_rels(self, data_path):
        qid2pid = dict()

        fin = open(data_path, 'r',encoding='utf8')

        line_num = 0
        while True:
            line_num += 1
            line = fin.readline()
            if not line:
                break
            qid, rels = line.split('\t')
            rels = rels.split('||')
            pid_list = []
            for pid_score in rels:
                items = pid_score.split(',')  # [a00000fa_0d421bc801544699675c7c32,4]
                pid = items[0]
                score = int(items[1])
                if score >= 3:
                    pid_list.append(pid) ## pid: str
            qid2pid[qid] = pid_list
        fin.close()

        return qid2pid
