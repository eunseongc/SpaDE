import math
import time
import torch

import numpy as np
import torch.nn as nn

STOPLIST = ["a", "about", "also", "am", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be",
            "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't",
            "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
            "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's",
            "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it",
            "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't",
            "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours",
            "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some",
            "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
            "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
            "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were",
            "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you",
            "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"]

printable = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
printableX = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. ')
printable3X = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- ')

printableD = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
printable3D = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')

STOPLIST_ = list(map(lambda s: ''.join(filter(lambda x: x in printable, s)), STOPLIST))

STOPLIST = {}
for w in STOPLIST_:
    STOPLIST[w] = True

def set_random_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getlocaltime():
    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())

def get_activation(act=None):
    if act == 'tanh':
        act = nn.Tanh()
    elif act == 'relu':
        act = nn.ReLU()
    elif act == 'softplus':
        act = nn.Softplus()
    elif act == 'rrelu':
        act = nn.RReLU()
    elif act == 'leakyrelu':
        act = nn.LeakyReLU()
    elif act == 'elu':
        act = nn.ELU()
    elif act == 'selu':
        act = nn.SELU()
    elif act == 'glu':
        act = nn.GLU()
    else:
        print('Defaulting to tanh activations...')
        act = nn.Tanh()
    return act

def shuffle_ensuring(indices):
    new_indices = np.random.permutation(indices)
    for i, _ in enumerate(new_indices):
        while indices[i] == new_indices[i]:
            new_indices[i] = np.random.choice(indices, 1)

    return new_indices

def cleanD(s, join=True):
    s = [(x.lower() if x in printable3X else ' ') for x in s]
    s = [(x if x in printableX else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' . ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else w.replace('-', '') + ' ( ' + ' '.join(w.split('-')) + ' ) ') for w in s]
    s = ' '.join(s).split()

    return ' '.join(s) if join else s

def cleanQ(s, join=True):
    s = [(x.lower() if x in printable3D else ' ') for x in s]
    s = [(x if x in printableD else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else (' ' if len(min(w.split('-'), key=len)) > 1 else '').join(w.split('-'))) for w in s]
    s = ' '.join(s).split()
    s = [w for w in s if w not in STOPLIST]

    return ' '.join(s) if join else s

