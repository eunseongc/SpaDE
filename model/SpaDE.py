import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from time import time
from IPython import embed
from collections import defaultdict
from scipy.sparse import csc_matrix
from transformers import AdamW, AutoModelForMaskedLM, AutoTokenizer, AutoConfig

from base.BaseModel import BaseModel
from dataloader.DataBatcher import DataBatcher

from utils.Tool import cleanD

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# https://github.com/stanford-futuredata/ColBERT/blob/master/colbert/modeling/colbert.py
class SpaDE(BaseModel):
    def __init__(self, dataset, model_conf, device):
        super(SpaDE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        
        self.model_conf = model_conf
        self.device = device
        self.epoch_num = 0
        self.start_epoch = 1

        self.train_batch_size = model_conf['train_batch_size']
        self.accumulation_size = model_conf['accumulation_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['lr']
        self.reg = model_conf['reg']
        self.num_triplet_per_epoch = model_conf['num_triplet_per_epoch']

        self.components = model_conf['components']
        self.num_expand = model_conf['num_expand']
        
        self.pairwise = model_conf['pairwise']
        self.pointwise = model_conf['pointwise']
        self.lamb = model_conf['lamb']
        
        self.bert_model_name = model_conf['bert_model_name']
        self.max_seq_length = model_conf['max_seq_length']

        self.expand_method = model_conf['expand_method']
        self.duplicate_term = model_conf['duplicate_term']
        self.combine_method = model_conf['combine_method']

        self.clean_text = model_conf['clean_text']
        self.log_saturation = model_conf['log_saturation']
        self.use_context = model_conf['use_context']

        self.add_token = model_conf['add_token']
        self.build_model()

    def build_model(self):
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
        self.bert = AutoModelForMaskedLM.from_pretrained(self.bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_conf = AutoConfig.from_pretrained(self.bert_model_name)

        if self.use_context:
            self.term_score_layer1 = nn.Linear(self.bert_conf.hidden_size * 2, self.bert_conf.hidden_size)
        else:
            self.term_score_layer1 = nn.Linear(self.bert_conf.hidden_size, self.bert_conf.hidden_size)
        self.term_score_layer2 = nn.Linear(self.bert_conf.hidden_size, 1)
        self.term_score_layer = nn.Sequential(self.dropout, self.term_score_layer1, self.relu, self.term_score_layer2)
        torch.nn.init.normal_(self.term_score_layer1.weight, std=0.02)
        torch.nn.init.normal_(self.term_score_layer2.weight, std=0.02)
        torch.nn.init.zeros_(self.term_score_layer1.bias)
        torch.nn.init.zeros_(self.term_score_layer2.bias)

        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.reg)
        self.to(self.device)

    def forward(self, batch_doc_id, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, return_cls_expand=False):
        '''
        batch_doc_id: (batch, seq_len)
        batch_doc_attention: (batch, seq_len)
        batch_doc_vocab: (batch, seq_len)
        '''
        # BERT: (batch, seq_len, hidden_size)
        batch_doc = self.bert(batch_doc_id, attention_mask = batch_doc_attention, output_hidden_states=True)
        last_hidden_states = batch_doc.hidden_states[-1]

        logits = batch_doc.logits
        # remove [<PAD>]
        last_hidden_states = last_hidden_states * batch_doc_attention.unsqueeze(2)
        # logits w/o [<PAD>]
        logits = logits * batch_doc_attention.unsqueeze(2)

        if return_cls_expand:
            cls_expand = self.relu(logits[:, 0, :])
        else:
            cls_expand = None

        # Term expand
        if self.components in ['expand', 'all']:
            if self.expand_method == 'tokenwise_sum':
                expand_score = self.relu(logits).sum(dim=1)
            elif self.expand_method == 'tokenwise_mean':
                expand_score = self.relu(logits).mean(dim=1)
            elif self.expand_method == 'tokenwise_max':
                # expand_score = self.relu(logits).max(dim=1)[0]                    
                expand_score = torch.zeros(logits.shape[0], logits.shape[2]).to(self.device)
                for i, logit in enumerate(logits):
                    expand_score[i] = logit[:batch_doc_attention[i].sum()].max(0)[0]

            elif self.expand_method == 'cls':
                expand_score = self.relu(logits[:, 0, :])
            elif self.expand_method == 'pool':
                expand_score = self.relu(self.bert.cls(last_hidden_states.mean(dim=1)))
            elif self.expand_method == 'topk':
                if self.add_token:
                    topk_tokens = torch.zeros((logits.shape[0], logits.shape[1], self.dataset.vocab_size+1), device=logits.device)
                    token_indices = torch.arange(logits.shape[1])
                    for t in range(logits.shape[0]):
                        topk_tokens[t][token_indices, batch_doc_vocab[t]] = logits[t][token_indices, batch_doc_vocab[t]]
                        logits[t][token_indices, batch_doc_vocab[t]] = 0
                    topk_tokens = topk_tokens[:, :, :-1]
                topk = torch.zeros_like(logits)
                vals, indices = torch.topk(logits, k=self.num_expand, dim=2, largest=True, sorted=False)
                topk.scatter_(2, indices, self.relu(vals))
                if self.add_token:
                    expand_score = (topk + topk_tokens).max(dim=1)[0]
                else:
                    expand_score = topk.max(dim=1)[0]
            elif self.expand_method == 'tokenwise_max_topk':
                expand_score = self.relu(logits).max(dim=1)[0]
                if self.add_token:
                    topk_tokens = torch.zeros((logits.shape[0], self.bert_conf.vocab_size+1), device=logits.device)
                    for t in range(logits.shape[0]):
                        topk_tokens[t][batch_doc_vocab[t]] = expand_score[t][batch_doc_vocab[t]]
                        expand_score[t][batch_doc_vocab[t]] = 0
                    topk_tokens = topk_tokens[:, :-1]
                vals, indices = torch.topk(expand_score, k=self.num_expand, dim=1, largest=True, sorted=False)
                expand_score = torch.zeros_like(expand_score)
                for b in range(expand_score.size(0)):
                    expand_score[b, indices[b]] = vals[b]
                
                if self.add_token:
                    expand_score += topk_tokens

            if self.components == 'expand':
                if self.log_saturation:
                    expand_score = torch.log(expand_score+1)
                return cls_expand, expand_score
        
        # Term rewriting
        if self.components in ['reweight', 'all']:

            if self.use_context:
                if self.use_context == 'cls':
                    contexts = last_hidden_states[:, 0, :]
                elif self.use_context == 'max_pooling':
                    contexts = (last_hidden_states * batch_doc_attention.unsqueeze(2)).max(dim=1)[0]
                elif self.use_context == 'mean_pooling':
                    contexts = (last_hidden_states * batch_doc_attention.unsqueeze(2)).mean(dim=1)
                elif self.use_context == 'cls_distant':
                    contexts = last_hidden_states[:, 0, :]
                contexts = contexts.unsqueeze(1).expand(-1, last_hidden_states.shape[1], -1)
                term_score_seq = self.relu(self.term_score_layer(torch.cat([last_hidden_states, contexts], dim=2))).squeeze()
            else:
                # (batch, seq_len, hidden) -> (batch, seq_len)
                term_score_seq = self.relu(self.term_score_layer(last_hidden_states)).squeeze(-1)
            
            # (batch, seq_len) -> (batch, vocab)
            term_score = torch.zeros((last_hidden_states.shape[0], self.bert_conf.vocab_size+1), device=self.device)
            
            if self.components == 'all' and self.combine_method == 'sep':
                term_score[:, :-1] = expand_score

            if self.duplicate_term == 'first':
                for b in range(term_score.shape[0]):
                    term_score[b, batch_doc_vocab[b]] = term_score_seq[b]
            elif self.duplicate_term == 'max':
                for b in range(term_score.shape[0]):
                    term_score[b, batch_doc_vocab[b]] = term_score_seq[b]
                    for i, (k, v) in enumerate(batch_doc_vocab_map[b].items()): 
                        term_score[b, k] = term_score_seq[b, v].max()
            elif self.duplicate_term == 'mean':
                for b in range(term_score.shape[0]):
                    term_score[b, batch_doc_vocab[b]] = term_score_seq[b]
                    for i, (k, v) in enumerate(batch_doc_vocab_map[b].items()):
                        term_score[b, k] = term_score_seq[b, v].mean()

            term_score = term_score[:, :-1]

            if self.components =='all':
                if self.combine_method == 'max':
                    term_score = torch.cat((term_score, expand_score), 1).reshape(term_score.shape[0], 2, -1).max(1)[0]
                elif self.combine_method == 'mean':
                    for b in range(term_score.shape[0]):
                        doc_col = torch.unique(batch_doc_vocab[b])[1:]
                        term_score[b, doc_col] = (term_score[b, doc_col] + expand_score[b, doc_col])/2
            else:
                expand_score = None

            if self.log_saturation:
                term_score = torch.log(term_score+1)

            return cls_expand, term_score

    def train_model_per_batch(self, batch_targets, batch_query_bow, batch_pos_indices, batch_neg_indices, num_negatives=1):
        
        batch_size = batch_pos_indices.shape[0]

        if self.train_batch_size == self.accumulation_size:
            batch_loader = DataBatcher(np.arange(batch_size), batch_size=batch_size, drop_remain=False, shuffle=False)
        else:
            batch_loader = DataBatcher(np.arange(batch_size), batch_size=self.accumulation_size, drop_remain=False, shuffle=False)

        self.optimizer.zero_grad()

        for _, batch_idx in enumerate(batch_loader):
            batch_pos_indices_acc = batch_pos_indices[batch_idx]
            batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc = self._tokenize(batch_pos_indices_acc, self.max_seq_length)
            
            batch_pos_doc_acc = batch_pos_doc_acc.to(self.device)
            batch_pos_doc_attention_acc = batch_pos_doc_attention_acc.to(self.device)
            batch_pos_doc_vocab_acc = torch.LongTensor(batch_pos_doc_vocab_acc).to(self.device)
            
            pos_cls_expand, pos_doc = self.forward(batch_pos_doc_acc, batch_pos_doc_attention_acc, batch_pos_doc_vocab_acc, batch_pos_doc_vocab_map_acc, return_cls_expand=True)
            pos_score = (batch_query_bow[batch_idx] * pos_doc).sum(dim=1, keepdim=True)

            batch_neg_indices_acc = batch_neg_indices[batch_idx]
            if num_negatives == 1:
                batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc, self.max_seq_length)
                batch_neg_doc_acc = batch_neg_doc_acc.to(self.device)
                batch_neg_doc_attention_acc = batch_neg_doc_attention_acc.to(self.device)
                batch_neg_doc_vocab_acc = torch.LongTensor(batch_neg_doc_vocab_acc).to(self.device)
                _, neg_doc = self.forward(batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)
                neg_score = (batch_query_bow[batch_idx] * neg_doc).sum(dim=1, keepdim=True)
            else:
                neg_scores = []
                batch_neg_indices_acc_list = np.hsplit(batch_neg_indices_acc, batch_neg_indices_acc.shape[1])
                for i in range(num_negatives):
                    batch_neg_indices_acc = batch_neg_indices_acc_list[i].squeeze()
                    batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc = self._tokenize(batch_neg_indices_acc, self.max_seq_length)
                    batch_neg_doc_acc = batch_neg_doc_acc.to(self.device)
                    batch_neg_doc_attention_acc = batch_neg_doc_attention_acc.to(self.device)
                    batch_neg_doc_vocab_acc = torch.LongTensor(batch_neg_doc_vocab_acc).to(self.device)
                    _, neg_doc = self.forward(batch_neg_doc_acc, batch_neg_doc_attention_acc, batch_neg_doc_vocab_acc, batch_neg_doc_vocab_map_acc)
                    neg_score = (batch_query_bow[batch_idx] * neg_doc).sum(dim=1, keepdim=True)
                    neg_scores.append(neg_score)
                neg_score = torch.cat(neg_scores, dim=-1)
            
            # compute loss 
            if self.pairwise == "CE":
                # [(batch, 1), (batch, 1)] -> (batch, 2)
                pos_neg_score = torch.cat([pos_score, neg_score], dim=-1)
                loss = self.CE_loss(pos_neg_score, torch.zeros(pos_neg_score.shape[0], dtype=torch.long, device=self.device))
            elif self.pairwise == "Margin_MSE":
                batch_targets_acc = torch.Tensor(batch_targets[batch_idx]).to(self.device)
                target_pos, target_neg = batch_targets_acc[:, 0], batch_targets_acc[:, 1]
                labels = (target_pos-target_neg).view(-1, 1)
                loss = self.MSE_loss((pos_score-neg_score), labels)
                # loss = torch.mean(torch.pow((pos_score - neg_score) - (target_pos - target_neg),2))
            elif self.pairwise == "IBN":
                all_doc = torch.cat([pos_doc, neg_doc], dim=0)
                all_score = torch.mm(batch_query_bow[batch_idx], all_doc.transpose(0, 1)) # (batch, batch*2)
                labels = torch.tensor(range(len(all_score)), dtype=torch.long, device=self.device)
                loss = self.CE_loss(all_score, labels)
            else:
                loss = 0.0
            
            if self.pointwise:
                if self.pointwise == 'doc':
                    recon = F.softmax(pos_doc, dim=1)
                    loss += self.lamb * (-(batch_query_bow[batch_idx] * (recon+1e-10).log()).sum(dim=1).mean())
                elif self.pointwise == 'cls':
                    recon = F.softmax(pos_cls_expand, dim=1)
                    loss += self.lamb * (-(batch_query_bow[batch_idx] * (recon+1e-10).log()).sum(dim=1).mean())
                else:
                    print(f'> {self.pointwise} for self.pointwise is wrong, please check the argument')
               
            loss = loss * (batch_idx.shape[0]/batch_size)
            # backward
            loss.backward()
        embed()
        # step
        self.optimizer.step()
        return loss

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
        queries_path = os.path.join('data', self.dataset.data_name, f'queries_sp_{q_n_w_sp}.pkl')

        queries_cols, queries_values = self._make_sparse_col_value(queries, queries_path, q_n_w_sp)
        
        train_targets_temp = dataset.train_targets
        if train_targets_temp is None:
            train_targets_temp = dataset.train_q_indices
        num_negatives = 1

        if self.start_epoch != 1:
            print(f'YOU MUST BE TRAINING CONTINUALLY, THE SRART EPOCH NUM IS {self.start_epoch}')
        for epoch in range(self.start_epoch, num_epochs + 1):
            data_start = self.num_triplet_per_epoch*(epoch-1)
            data_end = self.num_triplet_per_epoch*(epoch)
            train_q_indices = np.array(dataset.train_q_indices[data_start:data_end]).astype(np.long)
            train_pos_indices = np.array(dataset.train_pos_indices[data_start:data_end])
            train_neg_indices = np.array(dataset.train_neg_indices[data_start:data_end])
            train_targets = np.array(train_targets_temp[data_start:data_end])
            
            self.train()
            
            epoch_loss = 0.0
            batch_loader = DataBatcher(np.arange(len(train_q_indices)), batch_size=self.train_batch_size)
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
                batch_query_sp = torch.sparse_coo_tensor(batch_query_indices_sp, batch_query_values, size=(len(batch_idx), self.dataset.vocab_size), device=self.device)
                batch_query_bow = batch_query_sp.to_dense()

                batch_targets = train_targets[batch_idx]

                batch_loss = self.train_model_per_batch(batch_targets, batch_query_bow, batch_pos_indices, batch_neg_indices, num_negatives=num_negatives)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))

            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % (epoch_loss/num_batches), 'train time=%.2f' % epoch_train_time]

            # ======================== Valid
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                valid_loss = self.predict('valid').mean()
                valid_score = evaluator.evaluate(self, dataset, 'valid')
                valid_score_str = [f"{early_stop.early_stop_measure}={valid_score[early_stop.early_stop_measure]:.4f}", \
                                   f"valid_loss={valid_loss:.4f}"]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    self.epoch_num = early_stop.best_epoch
                    logger.info(f'Early stop triggered, best epoch: {self.epoch_num} @ current epoch: {epoch}')
                    break
                elif updated:
                    torch.save(self.state_dict(), os.path.join(log_dir, f'{epoch}_best_model.p'))
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

            valid_targets = self.dataset.valid_targets
            if valid_targets is None:
                valid_targets = self.dataset.valid_q_indices
            valid_targets= valid_targets[:5000]

            queries = self.dataset.queries
            
            eval_loss = torch.zeros(len(valid_q_indices), device=self.device)

            batch_loader = DataBatcher(np.arange(len(valid_q_indices)), batch_size=self.test_batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc='valid..', dynamic_ncols=True)):
                batch_pos_indices = valid_pos_indices[batch_idx]
                batch_neg_indices = valid_neg_indices[batch_idx]
                batch_query_indices = valid_q_indices[batch_idx]
                batch_targets = valid_targets[batch_idx]

                # tokenize 
                batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map = self._tokenize(batch_pos_indices, self.max_seq_length)
                batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map = self._tokenize(batch_neg_indices, self.max_seq_length)

                batch_pos_doc_vocab = torch.LongTensor(batch_pos_doc_vocab).to(self.device)
                batch_neg_doc_vocab = torch.LongTensor(batch_neg_doc_vocab).to(self.device)

                batch_pos_doc = torch.LongTensor(batch_pos_doc).to(self.device)
                batch_pos_doc_attention = torch.LongTensor(batch_pos_doc_attention).to(self.device)
                batch_neg_doc = torch.LongTensor(batch_neg_doc).to(self.device)
                batch_neg_doc_attention = torch.LongTensor(batch_neg_doc_attention).to(self.device)
                
                # compute loss
                _, pos_doc = self.forward(batch_pos_doc, batch_pos_doc_attention, batch_pos_doc_vocab, batch_pos_doc_vocab_map)
                _, neg_doc = self.forward(batch_neg_doc, batch_neg_doc_attention, batch_neg_doc_vocab, batch_neg_doc_vocab_map)
 
                batch_query_bow = torch.Tensor(queries[batch_query_indices].toarray()).to(self.device)

                pos_score = (batch_query_bow * pos_doc).sum(dim=1, keepdim=True)
                neg_score = (batch_query_bow * neg_doc).sum(dim=1, keepdim=True)

                if self.pairwise == "CE":
                    # [(batch, 1), (batch, 1)] -> (batch, 2)
                    pos_neg_score = torch.cat([pos_score, neg_score], dim=-1)
                    batch_loss = self.CE_loss(pos_neg_score, torch.zeros(pos_neg_score.shape[0], dtype=torch.long, device=self.device))
                elif self.pairwise == "Margin_MSE":
                    batch_targets = torch.Tensor(batch_targets).to(self.device)
                    target_pos, target_neg = batch_targets[:, 0], batch_targets[:, 1]
                    labels = (target_pos-target_neg).view(-1, 1)
                    batch_loss = self.MSE_loss((pos_score-neg_score), labels)
                elif self.pairwise == "IBN":
                    all_pos_doc = pos_doc.repeat(pos_doc.shape[0], 1, 1) 
                    all_query_bow = batch_query_bow.unsqueeze(1).repeat(1, pos_doc.shape[0], 1)
                    all_pos_score = (all_query_bow * all_pos_doc).sum(dim=-1) # (batch, batch*2) 
                    pos_neg_score = torch.cat([all_pos_score, neg_score], dim=-1)
                    batch_loss = self.CE_loss(pos_neg_score, torch.arange(pos_neg_score.shape[0], dtype=torch.long, device=self.device))

                # eval_output[batch_idx] = recon.cpu()
                eval_loss[batch_idx[0]:batch_idx[-1]+1] = batch_loss

        return eval_loss.cpu().numpy()

    def get_token_to_vocab(self, doc_bert_ids, return_col=False):
        doc_vocab = np.zeros_like(doc_bert_ids)-1
        doc_col = []

        if self.bert_model_name in ['bert-base-uncased', 'bert-large-uncased']:
            for i, doc in enumerate(doc_bert_ids):
                doc_tokens = self.tokenizer.convert_ids_to_tokens(doc)
                doc_col.append([])
                full_token = ''
                full_token_start = 0
                full_tokens = []
                for j, token in enumerate(doc_tokens):
                    # [CLS]
                    if j == 0:
                        continue

                    # Tokens
                    if token.startswith('##'):
                        full_token += token[2:]
                    else:
                        if full_token != '':
                            # prior full token
                            full_word = self.bertword2word.get(full_token)
                            word_index = self.word2index.get(full_word)
                            if (word_index is not None) and (full_word not in full_tokens):
                                doc_vocab[i, full_token_start] = word_index
                                doc_col[-1].append(word_index)
                                full_tokens.append(full_word)

                        # cur token
                        full_token = token
                        full_token_start = j

                    # [SEP]
                    if token == '[SEP]':
                        full_word = self.bertword2word.get(full_token)
                        word_index = self.word2index.get(full_word)
                        if (word_index is not None) and (full_word not in full_tokens):
                            doc_vocab[i, full_token_start] = word_index
                            doc_col[-1].append(word_index)
                            full_tokens.append(full_word)
                        break
                    
            if return_col:
                return doc_vocab, doc_col
            return doc_vocab


    def get_token_to_vocab_bpe(self, doc_bert_ids, return_col=False):
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
                    if self.dataset.expand_collection:
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

        if return_col:
            return doc_vocab, doc_vocab_map, doc_col

        return doc_vocab, doc_vocab_map
 
    def restore(self, log_dir, epoch=None):
        if epoch is not None:
            self.epoch_num = epoch
        print(f"Restore model from the epoch {self.epoch_num}")
        with open(os.path.join(log_dir, f"{self.epoch_num}_best_model.p"), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def get_sparse_output(self, mode='test'):
        with torch.no_grad():          
            if mode == 'valid':
                input_pids = self.dataset.doc_id_valid
            elif mode == 'test':
                input_pids = self.dataset.doc_id

            rows, cols, values = [], [], []
            batch_doc_cols = []
        
            # cls_list = []
            
            output_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.epoch_num}_{len(input_pids)}_{self.num_expand}.pkl')
            # if os.path.exists(output_path):
            #     with open(output_path, 'rb') as f:
            #         return pickle.load(f)
            self.logger.info(f'Expand terms = {self.num_expand}')
            batch_loader = DataBatcher(np.arange(len(input_pids)), batch_size=self.test_batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc='Getting sparse output...', dynamic_ncols=True)):
                batch_indices = [input_pids[i] for i in batch_idx]
                batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self._tokenize(batch_indices, self.max_seq_length, return_col=True)
                batch_doc = torch.LongTensor(batch_doc).to(self.device)
                batch_doc_attention = torch.LongTensor(batch_doc_attention).to(self.device)
                batch_doc_vocab = torch.LongTensor(batch_doc_vocab).to(self.device)                
                                
                _, recon = self.forward(batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map)
                # cls_list.append(cls.cpu().numpy())

                if self.expand_method in ['topk', 'tokenwise_max_topk']:
                    row, col = torch.nonzero(recon, as_tuple=True)
                    value = recon[(row, col)]
                    if b % 1000 == 0:
                        print(row.shape)
                    rows.append(row.cpu().numpy() + b * batch_idx.shape[0])
                    cols.append(col.cpu().numpy())
                    values.append(value.cpu().numpy())

                    if b == 0:
                        print(f'expand_method is {self.expand_method} / {len(col.reshape(-1))} words in batch size {batch_idx.shape[0]}')

                else:
                    top_val_ori = []
                    top_col_ori = []
                    top_row_ori = []

                    # Original Terms (Rescoring)
                    for i in range(len(batch_idx)):
                        top_col_ori.append(batch_doc_col[i])
                        top_val_ori.append(recon[i, batch_doc_col[i]].cpu().numpy())
                        top_row_ori.append(np.full(len(batch_doc_col[i]), fill_value=batch_idx[i]))
                        recon[i, batch_doc_col[i]] = 0

                    rows += top_row_ori
                    cols += top_col_ori
                    values += top_val_ori
                    batch_doc_cols += batch_doc_col

                    top_val_inj = np.array([])
                    
                    # Inject Terms (Add scoring)
                    if self.components in ['all', 'expand']:
                        top_val_inj, top_col_inj = recon.topk(self.num_expand, sorted=False)
                        top_row_inj = batch_idx.reshape(-1,1).repeat(self.num_expand, axis=1).reshape(-1)
                    
                        rows.append(top_row_inj)
                        cols.append(top_col_inj.cpu().numpy().reshape(-1))
                        values.append(top_val_inj.cpu().numpy().reshape(-1))
 
                    if b == 0:
                        print(f'{len(top_val_inj.reshape(-1))} words for inject in batch size {batch_idx.shape[0]}')

            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            values = np.concatenate(values)

            output = csc_matrix((values, (rows, cols)), shape=(len(input_pids), self.dataset.vocab_size))
            print(f'{output.shape} shpae of sparse matrix is created')

            with open(output_path, 'wb') as f:
                pickle.dump(output, f, protocol=4)
            # with open(os.path.join(self.logger.log_dir, f'sparse_output_{self.epoch_num}_{len(input_pids)}_cls.pkl'), 'wb') as f:
            #     pickle.dump(cls_list, f, protocol=4)

        return output

    def _make_sparse_col_value(self, matrix, path, n_w_sp):
        if os.path.exists(path):
            print(path, "loaded!")
            with open(path, 'rb') as f:
                cols, values = pickle.load(f)
        else:
            cols = np.zeros((matrix.shape[0], n_w_sp))
            values = np.zeros((matrix.shape[0], n_w_sp))
            for i, doc in enumerate(tqdm(matrix, desc="> Converting sparse matrix into index-value matrix ...")): ## for every document, 
                leng = doc.nnz ## number of words
                cols[i, :leng] = doc.indices ## 
                values[i, :leng] = doc.data

            with open(path, 'wb') as f:
                pickle.dump((cols, values), f, protocol=4)
            print(path, "saved!")

        return cols, values

    def _tokenize(self, batch_indices, max_seq_length, return_col=False):
        # get text from indices
        batch_doc_text = [self.dataset.passage2text[str(i)] for i in batch_indices]


        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=max_seq_length, return_tensors='pt')
        batch_doc, batch_doc_attention = batch_doc_token['input_ids'], batch_doc_token['attention_mask']
        batch_doc_token = self.tokenizer(batch_doc_text, padding=True, truncation=True, max_length=max_seq_length)

        if return_col:
            batch_doc_vocab, batch_doc_vocab_map, batch_doc_col = self.get_token_to_vocab_bpe(batch_doc_token['input_ids'], return_col=return_col)
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map, batch_doc_col
        else:
            batch_doc_vocab, batch_doc_vocab_map = self.get_token_to_vocab_bpe(batch_doc_token['input_ids'])
            return batch_doc, batch_doc_attention, batch_doc_vocab, batch_doc_vocab_map