import numpy as np
from collections import OrderedDict
from dataloader.DataBatcher import DataBatcher


class Evaluator:
    def __init__(self, early_stop_measure='mrr_10', ks=[10, 1000], semantic_eval=False, batch_eval=False, sparse_eval=False, threshold=0.005):
        """
        :param int ks: top-k values to compute.
        """
        self.ks = ks
        self.max_k = max(self.ks)
        self.early_stop_measure = early_stop_measure

    def evaluate(self, model, dataset, mode='valid'):
        if mode == 'valid':
            eval_matrix = dataset.valid_matrix.toarray()
            eval_id = dataset.valid_id
            qid2pid = dataset.qid2pid_valid
            doc_id = dataset.doc_id_valid
        elif 'test' in mode:
            eval_matrix = dataset.test_matrix.toarray()
            eval_id = dataset.test_id
            qid2pid = dataset.qid2pid
            doc_id = dataset.doc_id

        score = OrderedDict()
        topns = self.ks
        # make prediction and compute scores
        closest_docs = get_closest_docs(eval_matrix.astype('float32'), model, method='dot', mode=mode) ## method: doc, cosine, gating
        
        ir_score = retrieval_evaluate(closest_docs, doc_id, eval_id, qid2pid, topns)
        for i, topn in enumerate(topns):
            score[f'prec_{topn}'], score[f'recall_{topn}'], score[f'mrr_{topn}'] = ir_score[0][i], ir_score[1][i], ir_score[2][i]
        score['MAP'] = ir_score[3]

        return score

def get_closest_docs(query_vectors, model, method='cosine', mode='test'):
    
    document_length = 8841823
    query_length = query_vectors.shape[0]
    # print(f'document_length: {document_length}')
    # print(f'query_vectors.shape: {query_vectors.shape}')

    # return --> matrix of Top-N documents by each query = (n_query, 1000)
    topN = np.zeros((query_length, 1000), dtype=np.int32)

    if method == 'dot':
        document_output = model.get_sparse_output(mode)

        batch_loader = DataBatcher(np.arange(query_length), batch_size=1, drop_remain=False, shuffle=False)
        for b, (batch_idx) in enumerate(batch_loader):
            batch_query = query_vectors[batch_idx] ## (1, num_docs)'
            query_indices = np.nonzero(batch_query[0])[0]
            batch_score = document_output[:,query_indices].sum(1).T
            topN[batch_idx] = getTopN(batch_score)
    else:
        raise("get_closest_docs()의 method를 정확히 입력하세요.")

    return topN

def retrieval_evaluate_valid(closest_docs, document_ids, query_ids, query2document, valid_measure, valid_k):
    # closest_docs     = [num_query, TOPN]
    # document_ids     = [num_document]           : pid 저장
    # query_ids        = [num_query]              : qid 저장
    # query2document   = {qid : pid_list}

    tmp_closest = closest_docs[:, :valid_k]
    measures = []

    # For each query
    for i, qid in enumerate(query_ids):
        # predict_pid
        pred_pid = [document_ids[p] for p in tmp_closest[i]]
        # answer_pid
        true_pid = query2document[qid]
        if len(true_pid) == 0:
            continue

        if 'prec' in valid_measure:
            measures.append(preck(true_pid, pred_pid, valid_k))
        elif 'recall' in valid_measure:
            measures.append(recallk(true_pid, pred_pid, valid_k)) 
        elif 'mrr' in valid_measure:
            measures.append(mrrk(true_pid, pred_pid, valid_k))        
        elif 'map' in valid_measure:
            measures.append(apk(true_pid, pred_pid, valid_k))
        else:
            raise("Incorrect earlystop measure")


    return np.mean(measures)

def retrieval_evaluate(closest_docs, document_ids, query_ids, query2document, topn_list=[1, 3, 5]):
    # closest_docs     = [num_query, 1000]
    # document_ids     = [num_document]           : pid 저장
    # query_ids        = [num_query]              : qid 저장
    # query2document   = {qid : pid_list}

    precisions = []
    recalls = []
    MRRs = []

    ## Get Precisions & Recall & MRR
    for n_docs in topn_list: 
        tmp_closest = closest_docs[:, :n_docs]

        precisions_k = []
        recalls_k = []
        MRRs_k = []
        # For each query
        for i, qid in enumerate(query_ids):
            if not query2document.get(qid):
                continue
            # predict_pid
            pred_pid = [str(document_ids[p]) for p in tmp_closest[i]] ## [10, 1, 2, 3] documents_ids[10]: 
            # answer_pid
            true_pid = query2document[qid] ## 7067032
            if len(true_pid) == 0:
                continue

            precisions_k.append(preck(true_pid, pred_pid, n_docs))
            recalls_k.append(recallk(true_pid, pred_pid, n_docs))
            MRRs_k.append(mrrk(true_pid, pred_pid, n_docs))

        precisions.append(np.mean(precisions_k))
        recalls.append(np.mean(recalls_k))
        MRRs.append(np.mean(MRRs_k))
    
    ## Get MAP
    # For each query
    MAP_k = []
    for i, qid in enumerate(query_ids):
        if not query2document.get(qid):
            continue
        # predict_pid
        pred_pid = [str(document_ids[p]) for p in closest_docs[i]]
        # answer_pid
        true_pid = query2document[qid]
        if len(true_pid) == 0:
            continue

        MAP_k.append(apk(true_pid, pred_pid, 1000))

    MAP = np.mean(MAP_k)

    return precisions, recalls, MRRs, MAP

def preck(actual, predicted, k=10):
    # actual      = A list of elements that are to be predicted (order doesn't matter)
    # predicted   = A list of predicted elements (order does matter)
    # k           = The maximum number of predicted elements

    return len(set(predicted).intersection(actual)) / len(predicted)

def recallk(actual, predicted, k=10):
    # actual      = A list of elements that are to be predicted (order doesn't matter)
    # predicted   = A list of predicted elements (order does matter)
    # k           = The maximum number of predicted elements
    # if np.random.random() < 0.02:
    #     relevant_pairs = set(predicted).intersection(actual)
    #     print(relevant_pairs)
    return len(set(predicted).intersection(actual)) / len(actual)

def mrrk(actual, predicted, k=10):
    # actual      = A list of elements that are to be predicted (order doesn't matter)
    # predicted   = A list of predicted elements (order does matter)
    # k           = The maximum number of predicted elements
    score = 0

    for i, p in enumerate(predicted):
        if p in actual:
            score = 1 / (i+1)
            break

    return score

def apk(actual, predicted, k=10):
    # actual      = A list of elements that are to be predicted (order doesn't matter)
    # predicted   = A list of predicted elements (order does matter)
    # k           = The maximum number of predicted elements
    score = 0
    num_hits = 0

    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1
            score += num_hits / (i+1)

    return score / min(len(actual), k)

def getTopN(batch_score):
    
    # TopN_indicies (not sorted)
    topN_part_indices = np.argpartition(-batch_score, kth=1000, axis=1)[:, :1000]

    # TopN_value (not sorted)
    topN_part = np.take_along_axis(batch_score, topN_part_indices, axis=1)

    # TopN_indices (sorted)
    topN_indicies = np.take_along_axis(topN_part_indices, np.argsort(-topN_part, axis=1), axis=1)

    return topN_indicies
