import os
import pickle
from statistics import mode
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from time import time
from IPython import embed
from collections import defaultdict
from scipy.sparse import csc_matrix
from transformers import AutoTokenizer, AutoConfig

from base.BaseModel import BaseModel
from dataloader.DataBatcher import DataBatcher

from utils import Config

class SpaDE_cotraining(BaseModel):
    def __init__(self, dataset, model_conf, device):
        super(SpaDE_cotraining, self).__init__(dataset, model_conf)
        self.dataset = dataset

        self.model_conf = model_conf
        self.device = device
        self.cur_iter = model_conf['start_iter']
        self.max_iter = model_conf['max_iter']
        self.sample_size = model_conf['sample_size']
        self.valid_no_training = model_conf['valid_no_training']

        self.train_batch_size = model_conf['train_batch_size']
        self.accumulation_size = model_conf['accumulation_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr1 = model_conf['lr1']
        self.lr2 = model_conf['lr2']

        self.reg = model_conf['reg']

        self.num_expand = model_conf['num_expand']

        self.pairwise = model_conf['pairwise']
        self.pointwise1 = model_conf['pointwise1']
        self.pointwise2 = model_conf['pointwise2']
        self.lamb = model_conf['lamb']

        self.bert_model_name = model_conf['bert_model_name']
        self.max_seq_length = model_conf['max_seq_length']

        self.path1 = model_conf['path1']  # SpaDE_weight
        self.num_epoch1 = model_conf['num_epoch1']
        self.path2 = model_conf['path2']  # SpaDE_expand
        self.num_epoch2 = model_conf['num_epoch2']
        self.alpha = model_conf['alpha']

        self.train_model1 = model_conf['train_model1']
        self.train_model2 = model_conf['train_model2']

        # set teaching_ratio or teaching_threshold
        self.teaching_start_iter = model_conf['teaching_start_iter']
        self.teaching_ratio = model_conf['teaching_ratio']
        self.teaching_threshold = model_conf['teaching_threshold']
        self.teaching_exclusive = model_conf['teaching_exclusive']
        self.teaching_loss_prob = model_conf['teaching_loss_prob']

        self.sparsifying_start_iter = model_conf['sparsifying_start_iter']
        self.sparsifying_step = model_conf['sparsifying_step']
        if self.sparsifying_start_iter < self.sparsifying_step:
            self.sparsifying_start_iter = self.sparsifying_step

        self.sparsifying_ratio = model_conf['sparsifying_ratio']
        self.sparsifying_method = model_conf['sparsifying_method']
        self.sparsifying_gradual = model_conf['sparsifying_gradual']

        self.expand_method2 = model_conf['expand_method2']
        self.num_expand2 = model_conf['num_expand2']

        self.topk_start_iter2 = model_conf['topk_start_iter2']

        self.need_expanded_doc = False

        if not (self.path1 and self.path2):
            print(f'Please input 2 paths properly')
            exit(1)

        self.build_model()

    def build_model(self):
        self.relu = nn.ReLU()
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_conf = AutoConfig.from_pretrained(self.bert_model_name)

        import model
        print("> Loading model1 ...")
        self.config1 = Config(main_conf_path=self.path1, model_conf_path=self.path1)
        model_name1 = self.config1.get_param('Experiment', 'model_name')
        MODEL_CLASS1 = getattr(model, model_name1)
        self.model1 = MODEL_CLASS1(self.dataset, self.config1['Model'], self.device)
        self.optimizer1 = self.model1.optimizer
        print(f"Model1 is initialized randomly")
        if self.optimizer1.param_groups[0]['lr'] != self.lr1:
            self.optimizer1.param_groups[0]['lr'] = self.lr1
            print(f"Set model1 learning rate to {self.lr1}")

        print("> Loading model2 ...")
        self.config2 = Config(main_conf_path=self.path2, model_conf_path=self.path2)
        model_name2 = self.config2.get_param('Experiment', 'model_name')
        MODEL_CLASS2 = getattr(model, model_name2)
        self.model2 = MODEL_CLASS2(self.dataset, self.config2['Model'], self.device)
        self.optimizer2 = self.model2.optimizer
        print(f"Model2 is initialized randomly")
        if self.optimizer2.param_groups[0]['lr'] != self.lr2:
            self.optimizer2.param_groups[0]['lr'] = self.lr2
            print(f"Set model2 learning rate to {self.lr2}")

        ## To check if we need expanded documents for input 
        if self.config1.get_param('Dataset', 'expand_collection') or self.config2.get_param('Dataset', 'expand_collection'):
            self.need_expanded_doc = True

        self.df_pruning_mask = np.ones(len(self.tokenizer.vocab))

        self.CE_loss = nn.CrossEntropyLoss()

        self.to(self.device)

    def forward(self, batch_doc_indices, batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map):
        # (batch_size, vocab_size)

        if self.config1.get_param('Dataset', 'expand_collection'):
            _, term_score1 = self.model1(batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)  # SpaDE_wegiht
        else:
            _, term_score1 = self.model1(batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map)  # SpaDE_weight no_exp

        if self.config2.get_param('Dataset', 'expand_collection'):
            _, term_score2 = self.model2(batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)  # SpaDE_expand
        else:
            _, term_score2 = self.model2(batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map)  # SpaDE_expand no_exp

        return term_score1, term_score2

    def forward_pos_neg(self, model, batch_query_bow,
                        batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map,
                        batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map,):

        # (batch_size, vocab_size)
        _, pos_doc = model(batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map)
        _, neg_doc = model(batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map)

        pos_score = (batch_query_bow * pos_doc).sum(dim=1, keepdim=True) # (batch_size, 1)
        neg_score = (batch_query_bow * neg_doc).sum(dim=1, keepdim=True) # (batch_size, 1)
        pos_neg_score = torch.cat([pos_score, neg_score], dim=1)
        pos_neg_prob = torch.softmax(pos_neg_score, dim=1)

        return pos_neg_prob

    def sampling(self, model, sample_query_bow, sample_pos_indices, sample_neg_indices, expand_collection=False):
        batch_loader = DataBatcher(np.arange(self.sample_size), batch_size=self.test_batch_size, drop_remain=False, shuffle=False)
        model.eval()
        sample_indices = []
        for _, batch_idx in enumerate(batch_loader):
            batch_query_bow = sample_query_bow[batch_idx].to(self.device)
            batch_pos_indices = sample_pos_indices[batch_idx]
            batch_neg_indices = sample_neg_indices[batch_idx]
            batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map = self._tokenize(batch_pos_indices, exp=expand_collection)
            batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map = self._tokenize(batch_neg_indices, exp=expand_collection)
            with torch.no_grad():
                pos_neg_prob = self.forward_pos_neg(model, batch_query_bow,
                                                    batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map,
                                                    batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map)
            neg_prob = pos_neg_prob[:, 1]
            sample_indices.append(neg_prob)

        neg_prob = torch.cat(sample_indices)
        _, sample_indices = torch.topk(neg_prob, k=max(1, int(self.sample_size*self.teaching_ratio)), largest=True, sorted=False)
        
        return sample_indices.cpu().numpy()

    def train_model_per_batch(self, model, optimizer, pointwise, batch_query_bow, batch_pos_indices, batch_neg_indices, expand_collection=False):
        batch_size = batch_pos_indices.shape[0]
        
        # ----------------------------- Train model1 -----------------------------
        optimizer.zero_grad()
        model.train()

        if self.train_batch_size == self.accumulation_size:
            batch_loader = DataBatcher(np.arange(batch_size), batch_size=batch_size, drop_remain=False, shuffle=False)
        else:
            batch_loader = DataBatcher(np.arange(batch_size), batch_size=self.accumulation_size, drop_remain=False, shuffle=False)

        for i, batch_idx in enumerate(batch_loader):
            batch_query_bow_acc = batch_query_bow[batch_idx].to(self.device)
            batch_pos_indices_acc = batch_pos_indices[batch_idx]
            batch_neg_indices_acc = batch_neg_indices[batch_idx]
            batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc = self._tokenize(batch_pos_indices_acc, exp=expand_collection)
            batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc, exp=expand_collection)

            _, pos_doc = model(batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc)
            _, neg_doc = model(batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)

            if model.components == 'expand':
                self.df_array += pos_doc.bool().cpu().numpy().sum(axis=0)
                self.df_array += neg_doc.bool().cpu().numpy().sum(axis=0)

            batch_query_bow_acc = torch.tensor(self.df_pruning_mask, dtype=torch.long, device=self.device) * batch_query_bow_acc
            
            if pointwise == 'doc':
                recon = F.softmax(pos_doc, dim=1)
                loss = self.lamb * (-(batch_query_bow_acc * (recon+1e-10).log()).sum(dim=1).mean())
            else:
                loss = 0.0

            if self.pairwise == "IBN":
                all_doc = torch.cat([pos_doc, neg_doc], dim=0)
                all_score = torch.mm(batch_query_bow_acc, all_doc.transpose(0, 1))  # (batch, batch*2)
                labels = torch.tensor(range(len(all_score)), dtype=torch.long, device=self.device)
                loss += self.CE_loss(all_score, labels)
            else:
                exit(1)
            
            loss = loss * (batch_idx.shape[0]/batch_size) ## batch_size: batch_pos_indices.shape[0]
            loss.backward()

        optimizer.step()
        return loss.item()

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        model1_expand = self.config1.get_param('Dataset', 'expand_collection')
        model2_expand = self.config2.get_param('Dataset', 'expand_collection')

        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']

        log_dir = logger.log_dir
        self.logger = logger
        start = time()

        # for sparse query
        queries = dataset.queries
        q_n_w_sp = queries.astype('bool').sum(1).max()
        queries_path = os.path.join('data', self.dataset.data_name, f'queries_sp_{q_n_w_sp}.pkl')

        queries_cols, queries_values = self._make_sparse_col_value(queries, queries_path, q_n_w_sp)

        best_valid_score, best_valid_score_measure, should_stop = None, 0.0, False
        logger.info(f'> {self.sample_size/self.train_batch_size} iterations per sampling')
        logger.info(f'> Maximum sampling count: {(self.sample_size * self.max_iter)/self.train_batch_size} samplings')

        start_s_i = 0
        self.triples_num = 0
        self.df_array = np.zeros(len(self.tokenizer.vocab))
        self.cur_iter = int(self.cur_iter)
        self.sparsifying_cnt = 0

        for s_i in range(start_s_i, int((self.train_batch_size * self.max_iter)/self.sample_size)): ## sample_index
            if self.cur_iter == 0 and self.valid_no_training:
                print(f'valid_no_training is True, validation will be started')
                self.eval()
                self.make_sparse_output(mode='valid')
                valid_score_str = ["[NO TRAINING]"]
                for alpha in [0.0, 0.3, 0.4, 0.5, 1.0]:
                    self.cur_alpha = alpha
                    valid_score = evaluator.evaluate(self, dataset, 'valid')
                    valid_score_str += [f"{early_stop.early_stop_measure}(alpha={self.cur_alpha})={valid_score[early_stop.early_stop_measure]:.4f}"]
                logger.info(valid_score_str)

            data_start = self.sample_size*(s_i) ## 0 for 0-th sample index
            data_end = self.sample_size*(s_i+1)
            train_q_indices = np.array(dataset.train_q_indices[data_start:data_end]).astype(np.long)
            train_pos_indices = np.array(dataset.train_pos_indices[data_start:data_end])
            train_neg_indices = np.array(dataset.train_neg_indices[data_start:data_end])
            
            # batch_query_bow
            train_query_row = torch.arange(self.sample_size).reshape(-1, 1).repeat(1, q_n_w_sp)
            train_query_col = torch.LongTensor(queries_cols[train_q_indices])
            train_query_indices_sp = torch.cat([train_query_row.reshape(1, -1), train_query_col.reshape(1, -1)], dim=0)
            train_query_values = torch.Tensor(queries_values[train_q_indices]).reshape(-1)
            train_query_sp = torch.sparse_coo_tensor(train_query_indices_sp, train_query_values, size=(self.sample_size, self.bert_conf.vocab_size))
            train_query_bow = train_query_sp.to_dense()

            co_teaching = self.cur_iter >= self.teaching_start_iter
            if co_teaching:
                m1_indices = self.sampling(self.model2, train_query_bow, train_pos_indices, train_neg_indices, expand_collection=model2_expand) 
                m2_indices = self.sampling(self.model1, train_query_bow, train_pos_indices, train_neg_indices, expand_collection=model1_expand) 
            else:
                m1_indices, m2_indices = np.arange(self.sample_size), np.arange(self.sample_size)

            batch_loader = DataBatcher(np.arange(len(m1_indices)), batch_size=self.train_batch_size)
            for _, batch_idx in enumerate(batch_loader):
                iter_start_time = time()
                self.cur_iter += 1

                if self.cur_iter == self.topk_start_iter2:
                    prev_expand_method = self.model2.expand_method
                    if self.expand_method2:
                        self.model2.expand_method = self.expand_method2
                        print(f"Model2's expand_method = {self.model2.expand_method}")
                    if self.num_expand2:
                        self.model2.num_expand = self.num_expand2
                        print(f"Model2's num_expand = {self.model2.num_expand}")
                    print(f"> Expand method for model2 is changed from {prev_expand_method} to {self.model2.expand_method} from {self.cur_iter} iter(s)")
                    print(f"> Num_expand {self.num_expand} is changed to {self.model2.num_expand}")
                    self.num_expand = self.model2.num_expand

                self.triples_num += batch_idx.shape[0]
                # ------------------------- Sparsification ----------------------------- #
                do_sparsify = self.cur_iter > self.sparsifying_start_iter and self.cur_iter % self.sparsifying_step == 0 and self.sparsifying_ratio < 1.0 
                if do_sparsify:
                    sparsifying_start_time = time()
                    num_bf = self.df_pruning_mask.sum()

                    if self.sparsifying_method == 'df_cutoff_btw':
                        if self.sparsifying_cnt > 5:
                            print(">> Already sparsified 3 times ... Skip sparsifying")
                        else:
                            if self.sparsifying_gradual: # 1 -> 0.9 / 2 -> 0.9 - 0.05 * (2-1) / 3 -> 0.9 - 0.05 * (3-2)
                                cur_ratio = max(self.sparsifying_ratio, (self.sparsifying_ratio+0.2) - 0.05 * self.sparsifying_cnt) # start ratio=0.9, step=0.05
                                if cur_ratio != self.sparsifying_ratio:
                                    early_stop.initialize()
                                    best_valid_score_measure = 0.0
                            else:
                                if self.sparsifying_cnt == 0:
                                    early_stop.initialize()
                                    best_valid_score_measure = 0.0

                                cur_ratio = self.sparsifying_ratio
                            df_cutoff = 2 * self.triples_num * cur_ratio

                            cutoff_indices = np.where(self.df_array > df_cutoff)[0]
                            self.df_pruning_mask[cutoff_indices] = 0

                            logger.info(f"> [Sparsifying method: {self.sparsifying_method}, Sparsifying_step: {self.sparsifying_step}]")
                            logger.info(f"> {2 * self.sparsifying_step * self.train_batch_size }")
                            logger.info(f"> Max df: {self.triples_num * 2}, df_cutoff: {df_cutoff} for cur_ratio: {cur_ratio}")
                            logger.info(f"> Max value in the current df_array is {self.df_array.max()}")
                            self.sparsifying_cnt += 1
                    else:
                        print(f">> Invalid sparsifying method")
                        exit(1)

                    logger.info(f'> {self.cur_iter}th iteration, |V| {num_bf} vocabs -> {self.df_pruning_mask.sum()} vocabs ({num_bf-self.df_pruning_mask.sum()} sparsified) / time taken: {time()-sparsifying_start_time:.2f}s')
                if self.cur_iter % self.sparsifying_step == 0:
                    logger.info(f"> Initialize df_array and number of triples ({self.triples_num})")
                    self.triples_num = 0
                    self.df_array = np.zeros(len(self.tokenizer.vocab))
                    
                # ------------------------- Sparsification -----------------------------
                batch_m1_indices = m1_indices[batch_idx]
                batch_query_bow = train_query_bow[batch_m1_indices]
                batch_pos_indices = train_pos_indices[batch_m1_indices]
                batch_neg_indices = train_neg_indices[batch_m1_indices]
                loss1 = self.train_model_per_batch(self.model1, self.optimizer1, self.pointwise1, batch_query_bow, batch_pos_indices, batch_neg_indices, expand_collection=model1_expand)

                batch_m2_indices = m2_indices[batch_idx]
                batch_query_bow = train_query_bow[batch_m2_indices]
                batch_pos_indices = train_pos_indices[batch_m2_indices]
                batch_neg_indices = train_neg_indices[batch_m2_indices]
                loss2 = self.train_model_per_batch(self.model2, self.optimizer2, self.pointwise2, batch_query_bow, batch_pos_indices, batch_neg_indices, expand_collection=model2_expand)
                            
                if self.cur_iter % verbose == 0:
                    logger.info(f'> [s_i: {s_i}, iter: {self.cur_iter}, co-teaching: {co_teaching}, do_sparsify {do_sparsify}] {self.train_batch_size} triples/iter, {int(m1_indices.shape[0]/self.train_batch_size)} iters/sampling, {time() - iter_start_time:.2f}s/per_iter, loss = 1: {loss1:.3f} / 2: {loss2:.3f}')
                
                # ======================== Valid
                if (self.cur_iter >= test_from and self.cur_iter % test_step == 1) or self.cur_iter == self.max_iter:
                    self.eval()
                    # evaluate model
                    self.make_sparse_output(mode='valid')
                    valid_score_str = [f"iteration={self.cur_iter}"]

                    for alpha in [0.0, 0.3, 0.4, 0.5, 1.0]:
                        self.cur_alpha = alpha
                        valid_score = evaluator.evaluate(self, dataset, 'valid')
                        valid_score_str += [f"{early_stop.early_stop_measure}(alpha={self.cur_alpha})={valid_score[early_stop.early_stop_measure]:.4f}"]
                        if valid_score[early_stop.early_stop_measure] > best_valid_score_measure:
                            best_valid_score = valid_score
                            best_valid_score_measure = valid_score[early_stop.early_stop_measure]
                            self.alpha = alpha
                    logger.info(', '.join(valid_score_str))
                    valid_score = best_valid_score

                    updated, should_stop = early_stop.step(valid_score, self.cur_iter)

                    if should_stop:
                        self.cur_iter = early_stop.best_epoch
                        logger.info(f'Early stop triggered, best iterations: {self.cur_iter}')
                        break
                    elif updated:
                        torch.save(self.state_dict(), os.path.join(log_dir, f'{self.cur_iter}_best_model.p'))
                        torch.save(self.optimizer1.state_dict(), os.path.join(log_dir, f'{self.cur_iter}_best_optimizer1.p'))
                        torch.save(self.optimizer2.state_dict(), os.path.join(log_dir, f'{self.cur_iter}_best_optimizer2.p'))
                    
                        # Save df array and masking array
                        with open(os.path.join(log_dir, f'{self.cur_iter}_df_array.pkl'), 'wb') as f:
                            pickle.dump(self.df_array, f)
                        with open(os.path.join(log_dir, f'{self.cur_iter}_df_pruning_mask.pkl'), 'wb') as f:
                            pickle.dump(self.df_pruning_mask, f)
                    else:
                        pass

            if should_stop:
                break
                    
        total_train_time = start - time()
        return early_stop.best_score, total_train_time

    def restore(self, log_dir, cur_iter=None):
        if cur_iter is not None:
            self.cur_iter = cur_iter
        print(f"Restore model from the epoch {self.cur_iter}")
        # load model parameters
        with open(os.path.join(log_dir, f"{self.cur_iter}_best_model.p"), 'rb') as f:
            state_dict = torch.load(f)
            self.load_state_dict(state_dict)

        # load optimizers
        with open(os.path.join(log_dir, f"{self.cur_iter}_best_optimizer1.p"), 'rb') as f:
            opti1_state_dict = torch.load(f)
            self.optimizer1.load_state_dict(opti1_state_dict)
        with open(os.path.join(log_dir, f"{self.cur_iter}_best_optimizer2.p"), 'rb') as f:
            opti2_state_dict = torch.load(f)
            self.optimizer2.load_state_dict(opti2_state_dict)

        # load pruning mask and df array
        with open(os.path.join(log_dir, f"{self.cur_iter}_df_pruning_mask.pkl"), 'rb') as f:
            self.df_pruning_mask = pickle.load(f)
        with open(os.path.join(log_dir, f"{self.cur_iter}_df_array.pkl"), 'rb') as f:
            self.df_array = pickle.load(f)

    def get_sparse_output(self, mode='test'):
        if mode == 'valid':
            input_pids = self.dataset.doc_id_valid
            alpha = self.cur_alpha
        elif mode == 'test':
            input_pids = self.dataset.doc_id
            alpha = self.alpha
            self.logger.info(f"alpha: {self.alpha} (e.g., alpha * expand + (1-alpha) * weight")

        output_expand_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{self.num_expand}_{self.model2.expand_method}_expand.pkl')
        output_weight_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{self.num_expand}_weight.pkl')

        if not (os.path.exists(output_expand_path) and os.path.exists(output_weight_path)):
            output_weight, output_expand = self.make_sparse_output(mode=mode)
        else:
            with open(output_expand_path, 'rb') as f:
                output_expand = pickle.load(f)
            with open(output_weight_path, 'rb') as f:
                output_weight = pickle.load(f)

        output = output_weight * (1-alpha) + output_expand * alpha
        output = output.multiply(self.df_pruning_mask)
        output = output.tocsc()
        output.eliminate_zeros()

        return output

    def make_sparse_output(self, mode='test'):
        with torch.no_grad():
            self.eval()
            if mode == 'valid':
                input_pids = self.dataset.doc_id_valid
            elif mode == 'test':
                input_pids = self.dataset.doc_id
                self.logger.info(f'Expand terms = {self.num_expand}')

            rows_weight, cols_weight, values_weight = [], [], []
            rows_expand, cols_expand, values_expand = [], [], []
            batch_doc_cols = []

            print(f'> expand_method: {self.model2.expand_method} / num_expand: {self.model2.num_expand}')

            batch_loader = DataBatcher(np.arange(len(input_pids)), batch_size=self.test_batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc='Getting sparse output...', dynamic_ncols=True)):
                batch_indices = [input_pids[i] for i in batch_idx]
                batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self._tokenize(batch_indices, return_col=True)
                if self.need_expanded_doc:
                    batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map, batch_doc_exp_col = self._tokenize(batch_indices, return_col=True, exp=True)
                else:
                    batch_doc_exp, batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map, batch_doc_exp_col = None, None, None, None, None

                doc_weight, doc_expand = self.forward(batch_indices, batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_exp,
                                                      batch_doc_exp_attention, batch_doc_exp_vocab, batch_doc_exp_vocab_map)

                # ------------------------------------------------doc_expand start----------------------------------------------------
                top_val_ori, top_col_ori, top_row_ori = [], [], []
                
                if self.model2.expand_method in ['topk', 'tokenwise_max_topk']:
                    row, col = torch.nonzero(doc_expand, as_tuple=True)
                    value = doc_expand[(row, col)]
                    rows_expand.append(row.cpu().numpy() + b * self.test_batch_size)
                    cols_expand.append(col.cpu().numpy())
                    values_expand.append(value.cpu().numpy())

                    if b == 0:
                        print(f'expand_method is {self.model2.expand_method} / {len(col.reshape(-1))} words in batch size {batch_idx.shape[0]}')
                else:
                    if self.config2.get_param('Dataset', 'expand_collection'):
                        batch_doc_col = batch_doc_exp_col

                    # Original Terms (Rescoring)
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

                    if b == 0:
                        print(f'expand_method is {self.model2.expand_method} / {len(top_col_inj.reshape(-1))} words in batch size {batch_idx.shape[0]}')
                # -------------------------------------------------doc_expand end----------------------------------------------------

                # ------------------------------------------------doc_weight start----------------------------------------------------
                row, col = torch.nonzero(doc_weight, as_tuple=True)
                value = doc_weight[(row, col)]

                rows_weight.append(row.cpu().numpy() + b * self.test_batch_size)
                cols_weight.append(col.cpu().numpy())
                values_weight.append(value.cpu().numpy())
                # -------------------------------------------------doc_weight end----------------------------------------------------

            # concatenate all the arrays
            rows_expand = np.concatenate(rows_expand)
            cols_expand = np.concatenate(cols_expand)
            values_expand = np.concatenate(values_expand)

            rows_weight = np.concatenate(rows_weight)
            cols_weight = np.concatenate(cols_weight)
            values_weight = np.concatenate(values_weight)

            # create the sparse matrices
            output_expand = csc_matrix((values_expand, (rows_expand, cols_expand)), shape=(len(input_pids), self.bert_conf.vocab_size))
            output_expand = output_expand.multiply(self.df_pruning_mask)
            output_expand.eliminate_zeros()
            output_expand = output_expand.tocsc()

            output_weight = csc_matrix((values_weight, (rows_weight, cols_weight)), shape=(len(input_pids), self.bert_conf.vocab_size))
            output_weight = output_weight.multiply(self.df_pruning_mask)
            output_weight.eliminate_zeros()
            output_weight = output_weight.tocsc()

            print(f'{output_expand.shape} shpae of sparse matrix is created')
            with open(os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{self.num_expand}_{self.model2.expand_method}_expand.pkl'), 'wb') as f:
                pickle.dump(output_expand, f, protocol=4)
            with open(os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{self.num_expand}_weight.pkl'), 'wb') as f:
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
            batch_doc_text = [self.dataset.passage2text_exp[str(i)] for i in batch_indices]
        else:
            batch_doc_text = [self.dataset.passage2text[str(i)] for i in batch_indices]

        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors='pt')
        batch_doc, batch_doc_attention = batch_doc_token['input_ids'], batch_doc_token['attention_mask']
        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=self.max_seq_length)

        batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self.get_token_to_vocab_bpe(batch_doc_token['input_ids'], exp=exp, return_col=return_col)
        batch_doc, batch_doc_attention, batch_doc_vocab = batch_doc.to(self.device), batch_doc_attention.to(self.device), torch.LongTensor(batch_doc_vocab).to(self.device)

        if return_col:
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col
        else:
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map
    
