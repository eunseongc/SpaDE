import os
import pickle
from statistics import mode
import IPython
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from time import time
from IPython import embed
from collections import defaultdict
from scipy.sparse import csc_matrix, csr_matrix
from transformers import AdamW, AutoModelForMaskedLM, AutoTokenizer, AutoConfig

from base.BaseModel import BaseModel
from dataloader.DataBatcher import DataBatcher

from utils import Config

# https://github.com/stanford-futuredata/ColBERT/blob/master/colbert/modeling/colbert.py


class LASER_cotraining(BaseModel):
    def __init__(self, dataset, model_conf, device):
        super(LASER_cotraining, self).__init__(dataset, model_conf)
        self.dataset = dataset

        self.model_conf = model_conf
        self.device = device
        self.epoch_num = 0
        self.start_epoch = model_conf['start_epoch']

        self.sample_batch_size = model_conf['sample_batch_size']
        self.train_batch_size = model_conf['train_batch_size']
        self.accumulation_size = model_conf['accumulation_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['lr']
        self.reg = model_conf['reg']
        self.num_triplet_per_epoch = model_conf['num_triplet_per_epoch']

        self.components = model_conf['components']
        self.num_expand = model_conf['num_expand']

        self.pairwise = model_conf['pairwise']
        self.pointwise1 = model_conf['pointwise1']
        self.pointwise2 = model_conf['pointwise2']
        self.lamb = model_conf['lamb']

        self.bert_model_name = model_conf['bert_model_name']
        self.max_seq_length = model_conf['max_seq_length']

        self.path1 = model_conf['path1']  # LASER_weight
        self.num_epoch1 = model_conf['num_epoch1']
        self.path2 = model_conf['path2']  # LASER_expand
        self.num_epoch2 = model_conf['num_epoch2']
        self.alpha = model_conf['alpha']

        self.train_model1 = model_conf['train_model1']
        self.train_model2 = model_conf['train_model2']

        # set teaching_ratio or teaching_threshold
        self.teaching_start_epoch = model_conf['teaching_start_epoch']
        self.teaching_ratio = model_conf['teaching_ratio']
        self.teaching_threshold = model_conf['teaching_threshold']
        self.teaching_exclusive = model_conf['teaching_exclusive']

        if not (self.path1 and self.path2):
            print(f'Please input 2 paths properly')
            exit(1)

        self.build_model()

    def build_model(self):
        self.relu = nn.ReLU()
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_conf = AutoConfig.from_pretrained(self.bert_model_name)

        import model
        doc_id_backward = np.zeros(len(self.dataset.doc_id), dtype=np.int32)
        for i, pid in enumerate(tqdm(self.dataset.doc_id)):
            doc_id_backward[pid] = i
        print("> Loading model1 ...")
        if "sparse_output" in self.path1:
            print(f"> [NOTE] Model1 is not for training, you provide a sparse matrix for model1 from {self.path1}...")
            self.model1_matrix, self.train_model1 = True, False
            with open(f'{self.path1}', 'rb') as f:
                self.model1_output = pickle.load(f).tocsr()
                self.model1_output = self.model1_output[doc_id_backward]
        else:
            self.model1_matrix = False
            self.config1 = Config(main_conf_path=self.path1, model_conf_path=self.path1)
            model_name1 = self.config1.get_param('Experiment', 'model_name')
            MODEL_CLASS1 = getattr(model, model_name1)
            self.model1 = MODEL_CLASS1(self.dataset, self.config1['Model'], self.device)
            self.optimizer1 = self.model1.optimizer
            if self.num_epoch1:
                self.model1.restore(self.path1, epoch=self.num_epoch1)
                print(f"Loaded model1 from {self.path1}")
            else:
                print(f"Model1 is initialized randomly")

        print("> Loading model2 ...")
        if "sparse_output" in self.path2:
            print(f"> [NOTE] Model2 is not for training, you provide a sparse matrix for model2 from {self.path2}...")
            self.model2_matrix, self.train_model2 = True, False
            with open(f'{self.path2}', 'rb') as f:
                self.model2_output = pickle.load(f).tocsr()
                doc_id_backward = np.zeros(len(self.dataset.doc_id), dtype=np.int32)
                for i, pid in enumerate(tqdm(self.dataset.doc_id)):
                    doc_id_backward[pid] = i
                self.model2_output = self.model2_output[doc_id_backward]
        else:
            self.model2_matrix = False
            self.config2 = Config(main_conf_path=self.path2, model_conf_path=self.path2)
            model_name2 = self.config2.get_param('Experiment', 'model_name')
            MODEL_CLASS2 = getattr(model, model_name2)
            self.model2 = MODEL_CLASS2(self.dataset, self.config2['Model'], self.device)
            self.optimizer2 = self.model2.optimizer

            if self.num_epoch2:
                self.model2.restore(self.path2, epoch=self.num_epoch2)
                print(f"Loaded model2 from {self.path2}")
            else:
                print(f"Model2 is initialized randomly")

        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.num_accumulation1 = 0
        self.num_accumulation2 = 0

        self.to(self.device)

    def forward(self, batch_doc_indices, batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map):
        # (batch_size, vocab_size)

        if self.model1_matrix:
            term_score1 = self.model1_output[batch_doc_indices].toarray()
            term_score1 = torch.Tensor(term_score1).to(self.device)
        else:
            if self.config1.get_param('Dataset', 'expand_collection'):
                _, term_score1 = self.model1(batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)  # LASER_wegiht
            else:
                _, term_score1 = self.model1(batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map)  # LASER_weight no_exp

        if self.model2_matrix:
            term_score2 = self.model2_output[batch_doc_indices].toarray()
            term_score2 = torch.Tensor(term_score2).to(self.device)
        else:
            if self.config2.get_param('Dataset', 'expand_collection'):
                _, term_score2 = self.model2(batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)  # LASER_expand
            else:
                _, term_score2 = self.model2(batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map)  # LASER_expand no_exp
        return term_score1, term_score2

    def forward_pos_neg(self, model, batch_query_bow,
                        batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map,
                        batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map,):

        # (batch_size, vocab_size)
        _, pos_doc = model(batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map)
        pos_score = (batch_query_bow * pos_doc).sum(dim=1, keepdim=True) # (batch_size, 1)

        # (batch_size, vocab_size)
        _, neg_doc = model(batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map)
        neg_score = (batch_query_bow * neg_doc).sum(dim=1, keepdim=True) # (batch_size, 1)

        # [(batch, 1), (batch, 1)] -> (batch, 2)
        pos_neg_score = torch.cat([pos_score, neg_score], dim=1)
        pos_neg_prob = torch.softmax(pos_neg_score, dim=1)

        return pos_neg_prob

    def train_model_per_batch(self, batch_query_bow, batch_pos_indices, batch_neg_indices, epoch, num_negatives=1):
        batch_size = batch_pos_indices.shape[0]
        start_time = time()

        # ----------------------------- Get samples for each model -----------------------------
        if epoch < self.teaching_start_epoch or self.teaching_ratio >= 1.0:
            m1_indices = np.arange(batch_size)
            m2_indices = np.arange(batch_size)
        else:
            batch_loader = DataBatcher(np.arange(batch_size), batch_size=min(batch_size, self.test_batch_size), drop_remain=False, shuffle=False)
            if not self.model1_matrix:
                self.model1.eval()
            if not self.model2_matrix:
                self.model2.eval()

            m1_indices, m2_indices = [], []
            for _, batch_idx in enumerate(batch_loader):
                # Query
                batch_query_bow_acc = batch_query_bow[batch_idx]
                # Positive document
                batch_pos_indices_acc = batch_pos_indices[batch_idx]
                batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc = self._tokenize(batch_pos_indices_acc)
                batch_pos_doc_exp_acc, batch_pos_doc_exp_attention_acc, batch_pos_doc_exp_vocab_acc, batch_pos_doc_exp_vocab_map_acc = self._tokenize(batch_pos_indices_acc, exp=True)

                # Negative document
                batch_neg_indices_acc = batch_neg_indices[batch_idx]
                batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc)
                batch_neg_doc_exp_acc, batch_neg_doc_exp_attention_acc, batch_neg_doc_exp_vocab_acc, batch_neg_doc_exp_vocab_map_acc = self._tokenize(batch_neg_indices_acc, exp=True)
                
                # Get samples for training models
                with torch.no_grad():
                    # For training model 1
                    if self.train_model1:
                        if self.model2_matrix:
                            pos_doc, neg_doc = self.model2_output[batch_pos_indices_acc].toarray(), self.model2_output[batch_neg_indices_acc].toarray()
                            pos_doc, neg_doc = torch.Tensor(pos_doc).to(self.device), torch.Tensor(neg_doc).to(self.device)

                            pos_score = (batch_query_bow_acc * pos_doc).sum(dim=1, keepdim=True) # (batch_size, 1)
                            neg_score = (batch_query_bow_acc * neg_doc).sum(dim=1, keepdim=True) # (batch_size, 1)
                            pos_neg_score = torch.cat([pos_score, neg_score], dim=1)
                            pos_neg_prob_2 = torch.softmax(pos_neg_score, dim=1)
                        else:
                            if self.config2.get_param('Dataset', 'expand_collection'):
                                pos_neg_prob_2 = self.forward_pos_neg(self.model2, batch_query_bow_acc,
                                                                    batch_pos_doc_exp_acc, batch_pos_doc_exp_attention_acc, batch_pos_doc_exp_vocab_acc, batch_pos_doc_exp_vocab_map_acc,
                                                                    batch_neg_doc_exp_acc, batch_neg_doc_exp_attention_acc, batch_neg_doc_exp_vocab_acc, batch_neg_doc_exp_vocab_map_acc)
                            else:
                                pos_neg_prob_2 = self.forward_pos_neg(self.model2, batch_query_bow_acc,
                                                                    batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc,
                                                                    batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)
                        neg_prob_2 = pos_neg_prob_2[:, 1]
                            # Get sample indices for training models
                        if self.teaching_ratio > 0:
                            _, m1_indices_acc = torch.topk(neg_prob_2, k=max(1, int(self.test_batch_size*self.teaching_ratio)), largest=True, sorted=False)
                        elif self.teaching_threshold > 0:
                            m1_indices_acc = (neg_prob_2 > self.teaching_threshold).nonzero().view(-1)
                        m1_indices.append(m1_indices_acc + batch_idx[0])

                    # For training model 2
                    if self.train_model2:
                        if self.model1_matrix:
                            pos_doc, neg_doc = self.model1_output[batch_pos_indices_acc].toarray(), self.model1_output[batch_neg_indices_acc].toarray()
                            pos_doc, neg_doc = torch.Tensor(pos_doc).to(self.device), torch.Tensor(neg_doc).to(self.device)

                            pos_score = (batch_query_bow_acc * pos_doc).sum(dim=1, keepdim=True) # (batch_size, 1)
                            neg_score = (batch_query_bow_acc * neg_doc).sum(dim=1, keepdim=True) # (batch_size, 1)
                            pos_neg_score = torch.cat([pos_score, neg_score], dim=1)
                            pos_neg_prob_1 = torch.softmax(pos_neg_score, dim=1)
                        else:
                            if self.config1.get_param('Dataset', 'expand_collection'):
                                pos_neg_prob_1 = self.forward_pos_neg(self.model1, batch_query_bow_acc,
                                                                    batch_pos_doc_exp_acc, batch_pos_doc_exp_attention_acc, batch_pos_doc_exp_vocab_acc, batch_pos_doc_exp_vocab_map_acc,
                                                                    batch_neg_doc_exp_acc, batch_neg_doc_exp_attention_acc, batch_neg_doc_exp_vocab_acc, batch_neg_doc_exp_vocab_map_acc)
                            else:
                                pos_neg_prob_1 = self.forward_pos_neg(self.model1, batch_query_bow_acc,
                                                                    batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc,
                                                                    batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)
                        neg_prob_1 = pos_neg_prob_1[:, 1]
                        # Get sample indices for training models
                        if self.teaching_ratio > 0:
                            _, m2_indices_acc = torch.topk(neg_prob_1, k=max(1, int(self.test_batch_size*self.teaching_ratio)), largest=True, sorted=False)
                        elif self.teaching_threshold > 0:
                            m2_indices_acc = (neg_prob_1 > self.teaching_threshold).nonzero().view(-1)
                        m2_indices.append(m2_indices_acc + batch_idx[0])
            
            # Concatenate indices
            m1_indices = torch.cat(m1_indices, dim=0).cpu().numpy() if self.train_model1 else np.array([])
            m2_indices = torch.cat(m2_indices, dim=0).cpu().numpy() if self.train_model2 else np.array([])

            # Exclude the samples that are exists in both sets
            num_excluded = 0
            if self.teaching_exclusive:
                intersect_m1m2 = np.intersect1d(m1_indices, m2_indices)
                m1_indices = np.setdiff1d(m1_indices, intersect_m1m2)
                m2_indices = np.setdiff1d(m2_indices, intersect_m1m2)
                num_excluded = len(intersect_m1m2)
            print(f'Get {len(m1_indices)} (Model 1), {len(m2_indices)} (Model 2) samples from {batch_size} samples ({num_excluded} samples excluded, Exclusive = {self.teaching_exclusive}))')
        # ----------------------------- Get samples for each model -----------------------------
        end_time = time()
        # print('Get samples for each model time: {:.4f}s'.format(end_time - start_time))

        start_time = time()
        # ----------------------------- Train model1 -----------------------------
        if len(m1_indices) > 0 and self.train_model1:
            self.optimizer1.zero_grad()
            self.model1.train()
            batch_loader = DataBatcher(m1_indices, batch_size=self.accumulation_size, drop_remain=False, shuffle=False)
            for _, batch_idx in enumerate(batch_loader):
                # train_model(self.model1, batch_query_bow_acc, batch_pos_indices_acc, batch_neg_indices_acc, expand_collection=self.config1.get_param('Dataset', 'expand_collection'))
                batch_query_bow_acc = batch_query_bow[batch_idx]
                
                batch_pos_indices_acc = batch_pos_indices[batch_idx]
                batch_neg_indices_acc = batch_neg_indices[batch_idx]

                if self.config1.get_param('Dataset', 'expand_collection'):
                    batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc = self._tokenize(batch_pos_indices_acc, exp=True)
                    batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc, exp=True)
                else:
                    batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc = self._tokenize(batch_pos_indices_acc)
                    batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc)

                _, pos_doc_1 = self.model1(batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc)
                pos_score_1 = (batch_query_bow_acc * pos_doc_1).sum(dim=1, keepdim=True)

                _, neg_doc_1 = self.model1(batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)
                neg_score_1 = (batch_query_bow_acc * neg_doc_1).sum(dim=1, keepdim=True)

                if self.pairwise == "CE":
                    # [(batch, 1), (batch, 1)] -> (batch, 2)
                    pos_neg_score_1 = torch.cat([pos_score_1, neg_score_1], dim=-1)
                    loss_1 = self.CE_loss(pos_neg_score_1, torch.zeros(pos_neg_score_1.shape[0], dtype=torch.long, device=self.device))
                elif self.pairwise == "IBN":
                    all_doc = torch.cat([pos_doc_1, neg_doc_1], dim=0)
                    all_score = torch.mm(batch_query_bow_acc, all_doc.transpose(0, 1))  # (batch, batch*2)
                    labels = torch.tensor(range(len(all_score)), dtype=torch.long, device=self.device)
                    loss_1 = self.CE_loss(all_score, labels)
                else:
                    loss_1 = 0.0

                if self.pointwise1:
                    if self.pointwise1 == 'doc':
                        recon_1 = F.softmax(pos_doc_1, dim=1)
                        loss_1 += self.lamb * (-(batch_query_bow_acc * (recon_1+1e-10).log()).sum(dim=1).mean())
                    else:
                        print(f'> {self.pointwise1} for self.pointwise1 is wrong, please check the argument')
                
                
                loss_1 = loss_1 * (batch_idx.shape[0]/batch_size)
                loss_1.backward()

                self.num_accumulation1 += len(batch_idx)
                # step if accumulation_size is reached
                if self.num_accumulation1 >= self.train_batch_size:
                    self.optimizer1.step()
                    self.optimizer1.zero_grad()
                    self.num_accumulation1 = 0
        else:
            loss_1 = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # ----------------------------- Train model1 -----------------------------
        end_time = time()
        # print('Train model1 time: {:.4f}s'.format(end_time - start_time))
        # print('number of trainable samples: {}'.format(len(m1_indices)))

        start_time = time()
        # ----------------------------- Train model2 -----------------------------
        if len(m2_indices) > 0 and self.train_model2:
            self.optimizer2.zero_grad()
            self.model2.train()
            batch_loader = DataBatcher(m2_indices, batch_size=self.accumulation_size, drop_remain=False, shuffle=False)
            for _, batch_idx in enumerate(batch_loader):
                batch_query_bow_acc = batch_query_bow[batch_idx]

                # positive document
                batch_pos_indices_acc = batch_pos_indices[batch_idx]
                batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc = self._tokenize(batch_pos_indices_acc)

                # negative document
                batch_neg_indices_acc = batch_neg_indices[batch_idx]
                batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc)

                _, pos_doc_2 = self.model2(batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc)
                pos_score_2 = (batch_query_bow_acc * pos_doc_2).sum(dim=1, keepdim=True)
                _, neg_doc_2 = self.model2(batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)
                neg_score_2 = (batch_query_bow_acc * neg_doc_2).sum(dim=1, keepdim=True)

                if self.pairwise == "CE":
                    # [(batch, 1), (batch, 1)] -> (batch, 2)
                    pos_neg_score_2 = torch.cat([pos_score_2, neg_score_2], dim=-1)
                    loss_2 = self.CE_loss(pos_neg_score_2, torch.zeros(pos_neg_score_2.shape[0], dtype=torch.long, device=self.device))
                elif self.pairwise == "IBN":
                    all_doc = torch.cat([pos_doc_2, neg_doc_2], dim=0)
                    all_score = torch.mm(batch_query_bow_acc, all_doc.transpose(0, 1))  # (batch, batch*2)
                    labels = torch.tensor(range(len(all_score)), dtype=torch.long, device=self.device)
                    loss_2 = self.CE_loss(all_score, labels)
                else:
                    loss_2 = 0.0

                if self.pointwise2:
                    if self.pointwise2 == 'doc':
                        recon_2 = F.softmax(pos_doc_2, dim=1)
                        loss_2 += self.lamb * (-(batch_query_bow_acc * (recon_2+1e-10).log()).sum(dim=1).mean())
                    else:
                        print(f'> {self.pointwise2} for self.pointwise2 is wrong, please check the argument')
                loss_2 = loss_2 * (batch_idx.shape[0]/batch_size)
                loss_2.backward()

                self.num_accumulation2 += len(batch_idx)
                # step if accumulation_size is reached
                if self.num_accumulation2 >= self.train_batch_size:
                    self.optimizer2.step()
                    self.optimizer2.zero_grad()
                    self.num_accumulation2 = 0
        else:
            loss_2 = torch.tensor(0.0, dtype=torch.float, device=self.device)
        # ----------------------------- Train model2 -----------------------------
        end_time = time()
        # print('Train model2 time: {:.4f}s'.format(end_time - start_time))
        # print('number of trainable samples: {}'.format(len(m2_indices)))

        return loss_1.item(), loss_2.item()

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        self.logger = logger
        start = time()

        # for sparse query
        queries = dataset.queries
        q_n_w_sp = queries.astype('bool').sum(1).max()
        queries_path = os.path.join('data', self.dataset.data_name, f'queries_{dataset.vector}_sp_{q_n_w_sp}.pkl')

        queries_cols, queries_values = self._make_sparse_col_value(queries, queries_path, q_n_w_sp)

        if dataset.triples in ['marco2', 'marco4']:
            num_negatives = int(dataset.triples[-1])
        else:
            num_negatives = 1

        if self.start_epoch != 1:
            print(f'YOU MUST BE TRAINING CONTINUALLY, THE SRART EPOCH NUM IS {self.start_epoch}')
        for epoch in range(self.start_epoch, num_epochs + 1):
            # if epoch == 1:
            #     self.eval()
            #     valid_loss = self.predict('valid').mean()
            #     self.make_sparse_output(mode='valid')
            #     valid_score_str = [f"[NO TRAINING] valid_loss={valid_loss:.4f}\n"]
            #     for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
            #         self.cur_alpha = alpha
            #         valid_score = evaluator.evaluate(self, dataset, 'valid')
            #         valid_score_str += [f"{early_stop.early_stop_measure}(alpha={self.cur_alpha})={valid_score[early_stop.early_stop_measure]:.4f}"]
            #     logger.info(valid_score_str)

            data_start = self.num_triplet_per_epoch*(epoch-1)
            data_end = self.num_triplet_per_epoch*(epoch)
            train_q_indices = np.array(dataset.train_q_indices[data_start:data_end]).astype(np.long)
            train_pos_indices = np.array(dataset.train_pos_indices[data_start:data_end])
            train_neg_indices = np.array(dataset.train_neg_indices[data_start:data_end])

            self.train()
            epoch_loss1, epoch_loss2 = 0.0, 0.0
            batch_loader = DataBatcher(np.arange(len(train_q_indices)), batch_size=self.sample_batch_size)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(tqdm(batch_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)):
                batch_query_indices = train_q_indices[batch_idx]
                batch_pos_indices, batch_neg_indices = train_pos_indices[batch_idx], train_neg_indices[batch_idx]

                # batch_query_bow
                batch_query_row = torch.arange(len(batch_idx)).reshape(-1, 1).repeat(1, q_n_w_sp)
                batch_query_col = torch.LongTensor(queries_cols[batch_query_indices])
                batch_query_indices_sp = torch.cat([batch_query_row.reshape(1, -1), batch_query_col.reshape(1, -1)], dim=0)
                batch_query_values = torch.Tensor(queries_values[batch_query_indices]).reshape(-1)
                batch_query_sp = torch.sparse_coo_tensor(batch_query_indices_sp, batch_query_values, size=(len(batch_idx), self.bert_conf.vocab_size), device=self.device)
                batch_query_bow = batch_query_sp.to_dense()

                batch_loss1, batch_loss2 = self.train_model_per_batch(batch_query_bow, batch_pos_indices, batch_neg_indices, epoch, num_negatives=num_negatives)
                epoch_loss1 += batch_loss1
                epoch_loss2 += batch_loss2

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f(model 1), %.4f(model 2)' % (b + 1, num_batches, batch_loss1, batch_loss2))

            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f(model 1), %.3f(model 2)' % (epoch_loss1/num_batches, epoch_loss2/num_batches), 'train time=%.2f' % epoch_train_time]

            # ======================== Valid
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                valid_loss = self.predict('valid').mean()
                self.make_sparse_output(mode='valid')
                valid_score_str = [f"valid_loss={valid_loss:.4f}\n"]
                
                best_valid_score = None
                best_valid_score_measure = 0.0
                for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
                    self.cur_alpha = alpha
                    valid_score = evaluator.evaluate(self, dataset, 'valid')
                    valid_score_str += [f"{early_stop.early_stop_measure}(alpha={self.cur_alpha})={valid_score[early_stop.early_stop_measure]:.4f}"]
                    if valid_score[early_stop.early_stop_measure] > best_valid_score_measure:
                        best_valid_score = valid_score
                        best_valid_score_measure = valid_score[early_stop.early_stop_measure]
                        self.alpha = alpha
                valid_score = best_valid_score

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    self.epoch_num = early_stop.best_epoch
                    logger.info(f'Early stop triggered, best epoch: {self.epoch_num} @ current epoch: {epoch}')
                    break
                elif updated:
                    self.epoch_num = epoch
                    torch.save(self.state_dict(), os.path.join(log_dir, f'{epoch}_best_model.p'))
                    if self.model1_matrix:
                        pass
                    else:
                        torch.save(self.optimizer1.state_dict(), os.path.join(log_dir, f'{epoch}_best_optimizer1.p'))
                    if self.model2_matrix:
                        pass
                    else:
                        torch.save(self.optimizer2.state_dict(), os.path.join(log_dir, f'{epoch}_best_optimizer2.p'))
                else:
                    pass
                
                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str

            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time

    def predict(self, mode='valid'):
        with torch.no_grad():
            valid_q_indices = self.dataset.valid_q_indices[:5000]
            valid_pos_indices = self.dataset.valid_pos_indices[:5000]
            valid_neg_indices = self.dataset.valid_neg_indices[:5000]

            queries = self.dataset.queries

            eval_loss = torch.zeros(len(valid_q_indices), device=self.device)

            batch_loader = DataBatcher(np.arange(len(valid_q_indices)), batch_size=self.test_batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc='valid..', dynamic_ncols=True)):
                batch_pos_indices = valid_pos_indices[batch_idx]
                batch_neg_indices = valid_neg_indices[batch_idx]
                batch_query_indices = valid_q_indices[batch_idx]

                batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map = self._tokenize(batch_pos_indices)
                batch_pos_doc_exp, batch_pos_doc_exp_attention, batch_pos_doc_exp_vocab, batch_pos_doc_exp_vocab_map = self._tokenize(batch_pos_indices, exp=True)

                batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map = self._tokenize(batch_neg_indices)
                batch_neg_doc_exp, batch_neg_doc_exp_attention, batch_neg_doc_exp_vocab, batch_neg_doc_exp_vocab_map = self._tokenize(batch_neg_indices, exp=True)

                pos_doc1, pos_doc2 = self.forward(batch_pos_indices, batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map,
                                                  batch_pos_doc_exp, batch_pos_doc_exp_attention, batch_pos_doc_exp_vocab, batch_pos_doc_exp_vocab_map)
                pos_doc = (1-self.alpha) * pos_doc1 + self.alpha * pos_doc2

                neg_doc1, neg_doc2 = self.forward(batch_neg_indices, batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map,
                                                  batch_neg_doc_exp, batch_neg_doc_exp_attention, batch_neg_doc_exp_vocab, batch_neg_doc_exp_vocab_map)

                neg_doc = (1-self.alpha) * neg_doc1 + self.alpha * neg_doc2
                # compute loss
                batch_query_bow = torch.Tensor(queries[batch_query_indices].toarray()).to(self.device)

                pos_score = (batch_query_bow * pos_doc).sum(dim=1, keepdim=True)
                neg_score = (batch_query_bow * neg_doc).sum(dim=1, keepdim=True)

                if self.pairwise == "CE":
                    # [(batch, 1), (batch, 1)] -> (batch, 2)
                    pos_neg_score = torch.cat([pos_score, neg_score], dim=-1)
                    batch_loss = self.CE_loss(pos_neg_score, torch.zeros(pos_neg_score.shape[0], dtype=torch.long, device=self.device))
                elif self.pairwise == "IBN":
                    all_pos_doc = pos_doc.repeat(pos_doc.shape[0], 1, 1)
                    all_query_bow = batch_query_bow.unsqueeze(1).repeat(1, pos_doc.shape[0], 1)
                    all_pos_score = (all_query_bow * all_pos_doc).sum(dim=-1)  # (batch, batch*2)
                    pos_neg_score = torch.cat([all_pos_score, neg_score], dim=-1)
                    batch_loss = self.CE_loss(pos_neg_score, torch.arange(pos_neg_score.shape[0], dtype=torch.long, device=self.device))

                # eval_output[batch_idx] = recon.cpu()
                eval_loss[batch_idx[0]:batch_idx[-1]+1] = batch_loss

        return eval_loss.cpu().numpy()

    def restore(self, log_dir, epoch=None):
        if epoch is not None:
            self.epoch_num = epoch
        print(f"Restore model from the epoch {self.epoch_num}")
        # load model parameters
        with open(os.path.join(log_dir, f"{self.epoch_num}_best_model.p"), 'rb') as f:
            state_dict = torch.load(f)
            self.load_state_dict(state_dict)
        # load optimizers
        opti1_path = os.path.join(log_dir, f"{self.epoch_num}_best_optimizer1.p")
        if os.path.exists(opti1_path):
            with open(opti1_path, 'rb') as f:
                opti1_state_dict = torch.load(f)
                self.optimizer1.load_state_dict(opti1_state_dict)
    
        opti2_path = os.path.join(log_dir, f"{self.epoch_num}_best_optimizer2.p")
        if os.path.exists(opti2_path):
            with open(opti2_path, 'rb') as f:
                opti2_state_dict = torch.load(f)
                self.optimizer2.load_state_dict(opti2_state_dict)

    def get_sparse_output(self, mode='test'):
        if mode == 'valid':
            input_pids = self.dataset.doc_id_valid
            alpha = self.cur_alpha
        elif mode == 'test':
            input_pids = self.dataset.doc_id
            alpha = self.alpha

        output_expand_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.epoch_num}_{len(input_pids)}_{self.num_expand}_expand.pkl')
        output_weight_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.epoch_num}_{len(input_pids)}_{self.num_expand}_weight.pkl')

        if not (os.path.exists(output_expand_path) and os.path.exists(output_weight_path)):
            output_weight, output_expand = self.make_sparse_output(mode=mode)
        else:
            with open(output_expand_path, 'rb') as f:
                output_expand = pickle.load(f)
            with open(output_weight_path, 'rb') as f:
                output_weight = pickle.load(f)

        output = output_weight * (1-alpha) + output_expand * alpha

        return output

    def make_sparse_output(self, mode='test'):
        with torch.no_grad():
            self.eval()
            if mode == 'valid':
                input_pids = self.dataset.doc_id_valid
            elif mode == 'test':
                input_pids = self.dataset.doc_id

            rows_weight, cols_weight, values_weight = [], [], []
            rows_expand, cols_expand, values_expand = [], [], []
            batch_doc_cols = []

            self.logger.info(f'Expand terms = {self.num_expand}')
            batch_loader = DataBatcher(np.arange(len(input_pids)), batch_size=self.test_batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc='Getting sparse output...', dynamic_ncols=True)):
                batch_indices = [int(input_pids[i]) for i in batch_idx]
                batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self._tokenize(batch_indices, return_col=True)
                batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map, batch_doc_exp_col = self._tokenize(batch_indices, return_col=True, exp=True)

                doc_weight, doc_expand = self.forward(batch_indices, batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_exp,
                                                      batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)
                # doc_weight = (1 - self.alpha) * doc_weight
                # doc_expand = self.alpha * doc_expand

                # ----------------------------------------------------doc_expand----------------------------------------------------
                # Original Terms (Rescoring)
                top_val_ori, top_col_ori, top_row_ori = [], [], []

                for i in range(len(batch_idx)):
                    top_col_ori.append(batch_doc_col[i])
                    top_val_ori.append(doc_expand[i, batch_doc_col[i]].cpu().numpy())
                    top_row_ori.append(np.full(len(batch_doc_col[i]), fill_value=batch_idx[i]))
                    doc_expand[i, batch_doc_col[i]] = 0

                rows_expand += top_row_ori
                cols_expand += top_col_ori
                values_expand += top_val_ori
                batch_doc_cols += batch_doc_col

                # Expand Terms (New scoring)
                top_val_inj = np.array([])
                top_val_inj, top_col_inj = doc_expand.topk(self.num_expand, sorted=False)
                top_row_inj = batch_idx.reshape(-1, 1).repeat(self.num_expand, axis=1).reshape(-1)

                rows_expand.append(top_row_inj)
                cols_expand.append(top_col_inj.cpu().numpy().reshape(-1))
                values_expand.append(top_val_inj.cpu().numpy().reshape(-1))
                # ----------------------------------------------------doc_expand----------------------------------------------------

                # ----------------------------------------------------doc_weight----------------------------------------------------
                row, col = torch.nonzero(doc_weight, as_tuple=True)
                value = doc_weight[(row, col)]

                rows_weight.append(row.cpu().numpy() + b * batch_idx.shape[0])
                cols_weight.append(col.cpu().numpy())
                values_weight.append(value.cpu().numpy())
                # ----------------------------------------------------doc_weight----------------------------------------------------

            # concatenate all the arrays
            rows_expand = np.concatenate(rows_expand)
            cols_expand = np.concatenate(cols_expand)
            values_expand = np.concatenate(values_expand)

            rows_weight = np.concatenate(rows_weight)
            cols_weight = np.concatenate(cols_weight)
            values_weight = np.concatenate(values_weight)

            # create the sparse matrices
            output_expand = csc_matrix((values_expand, (rows_expand, cols_expand)), shape=(len(input_pids), self.bert_conf.vocab_size))
            output_expand.eliminate_zeros()
            output_weight = csc_matrix((values_weight, (rows_weight, cols_weight)), shape=(len(input_pids), self.bert_conf.vocab_size))
            output_weight.eliminate_zeros()

            print(f'{output_expand.shape} shpae of sparse matrix is created')
            with open(os.path.join(self.logger.log_dir, f'sparse_output_{self.epoch_num}_{len(input_pids)}_{self.num_expand}_expand.pkl'), 'wb') as f:
                pickle.dump(output_expand, f, protocol=4)
            with open(os.path.join(self.logger.log_dir, f'sparse_output_{self.epoch_num}_{len(input_pids)}_{self.num_expand}_weight.pkl'), 'wb') as f:
                pickle.dump(output_weight, f, protocol=4)
        return output_weight, output_expand

    def _make_sparse_col_value(self, matrix, path, n_w_sp):
        if os.path.exists(path):
            print(path, "loaded!")
            with open(path, 'rb') as f:
                cols, values = pickle.load(f)
        else:
            cols = np.zeros((matrix.shape[0], n_w_sp))
            values = np.zeros((matrix.shape[0], n_w_sp))
            for i, doc in enumerate(tqdm(matrix, desc="> Converting sparse matrix into index-value matrix ...")):  # for every document,
                leng = doc.nnz  # number of words
                cols[i, :leng] = doc.indices
                values[i, :leng] = doc.data

            with open(path, 'wb') as f:
                pickle.dump((cols, values), f, protocol=4)
            print(path, "saved!")

        return cols, values

    def get_token_to_vocab_bpe(self, doc_bert_ids, exp=False, return_col=False):
        doc_vocab = np.zeros_like(doc_bert_ids)-1
        doc_col = []
        doc_vocab_map = []

        for i, doc in enumerate(doc_bert_ids):
            doc_tokens = self.tokenizer.convert_ids_to_tokens(doc)
            doc_col.append([])
            num_sep = 0
            tokens = set()
            vocab_map = defaultdict(list)
            for j, token in enumerate(doc_tokens):
                # [CLS]
                if j == 0:
                    continue
                # [SEP]
                if token == '[SEP]':
                    if exp:
                        if num_sep == 1:
                            break
                        num_sep += 1
                    else:
                        break

                # Tokens
                if token not in tokens:
                    word_index = doc[j]
                    doc_vocab[i, j] = word_index
                    doc_col[-1].append(word_index)
                    tokens.add(token)
                vocab_map[doc[j]].append(j)

            new_vocab = dict()
            for k, v in vocab_map.items():
                if len(v) > 1:
                    new_vocab[k] = v
            doc_vocab_map.append(new_vocab)

        return doc_vocab, doc_vocab_map, doc_col

    def _tokenize(self, batch_indices, return_col=False, exp=False):
        # get text from indices:
        if exp:
            batch_doc_text = [self.dataset.passage2text_exp[int(i)] for i in batch_indices]
        else:
            batch_doc_text = [self.dataset.passage2text[int(i)] for i in batch_indices]

        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors='pt')
        batch_doc, batch_doc_attention = batch_doc_token['input_ids'], batch_doc_token['attention_mask']
        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=self.max_seq_length)

        batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self.get_token_to_vocab_bpe(batch_doc_token['input_ids'], exp=exp, return_col=return_col)
        batch_doc, batch_doc_attention, batch_doc_vocab = batch_doc.to(self.device), batch_doc_attention.to(self.device), torch.LongTensor(batch_doc_vocab).to(self.device)

        if return_col:
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col
        else:
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map
    
    def _tokenize_given_text(self, batch_doc_text, maxlen, return_col=False, exp=False):
        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
        batch_doc, batch_doc_attention = batch_doc_token['input_ids'], batch_doc_token['attention_mask']
        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=maxlen)

        if return_col:
            batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self.get_token_to_vocab_bpe(batch_doc_token['input_ids'], exp=exp, return_col=return_col)
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col
        else:
            batch_doc_vocab, batch_doc_vocab_map = self.get_token_to_vocab_bpe(batch_doc_token['input_ids'], exp=exp)
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map

    def encode_sentence_bert(self, batch_sentences, batch_idx, maxlen, alpha):
        with torch.no_grad():
            rows, cols, values = [], [], []
            batch_doc_text = batch_sentences
            batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self._tokenize_given_text(batch_doc_text, maxlen, return_col=True)
            batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map, batch_doc_exp_col = self._tokenize_given_text(batch_doc_text, maxlen, return_col=True, exp=True)

            batch_doc = torch.LongTensor(batch_doc).to(self.device)
            batch_doc_attention = torch.LongTensor(batch_doc_attention).to(self.device)
            batch_doc_vocab = torch.LongTensor(batch_doc_vocab).to(self.device)

            batch_doc_exp = torch.LongTensor(batch_doc_exp).to(self.device)
            batch_doc_exp_attention = torch.LongTensor(batch_doc_exp_attention).to(self.device)
            batch_doc_exp_vocab = torch.LongTensor(batch_doc_exp_vocab).to(self.device)

            doc_weight, doc_expand = self.forward(batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_exp,
                                                    batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)

            doc_weight = (1 - alpha) * doc_weight
            doc_expand = alpha * doc_expand

            # ----------------------------------------------------doc_expand----------------------------------------------------
            # Original Terms (Rescoring)
            top_val_ori, top_col_ori, top_row_ori = [], [], []
            
            for i in range(len(batch_sentences)):
                top_col_ori.append(batch_doc_col[i])
                top_val_ori.append(doc_expand[i, batch_doc_col[i]].cpu().numpy())
                top_row_ori.append(np.full(len(batch_doc_col[i]), fill_value=batch_idx[i]))
                doc_expand[i, batch_doc_col[i]] = 0

            rows += top_row_ori
            cols += top_col_ori
            values += top_val_ori

            # Expand Terms (New scoring)
            top_val_inj = np.array([])
            top_val_inj, top_col_inj = doc_expand.topk(self.num_expand, sorted=False)
            top_row_inj = batch_idx.reshape(-1, 1).repeat(self.num_expand, axis=1).reshape(-1)

            rows.append(top_row_inj)
            cols.append(top_col_inj.cpu().numpy().reshape(-1))
            values.append(top_val_inj.cpu().numpy().reshape(-1))

            # ----------------------------------------------------doc_weight----------------------------------------------------
            row, col = torch.nonzero(doc_weight, as_tuple=True)
            value = doc_weight[(row, col)]

            rows.append(row.cpu().numpy() + batch_idx[0])
            cols.append(col.cpu().numpy())
            values.append(value.cpu().numpy())
            # ----------------------------------------------------doc_weight----------------------------------------------------
        return rows, cols, values
