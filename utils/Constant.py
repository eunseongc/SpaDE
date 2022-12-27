DATASETS = []
MODELS = []

DATASET_TO_TYPE = {
    'ml-100k': 'UIRT',
    'ml-1m': 'UIRT',
    'ml-10m': 'UIRT',
    'ml-20m': 'UIRT',
    'netflix': 'UIRT',
    'amusic': 'UIRT',
    'yelp': 'UIRT',
    'agames': 'UIRT'
}

DATASET_TO_SEPRATOR = {
    'ml-100k': '\t',
    'ml-1m': '::',
    'ml-10m': '::',
    'ml-20m': ',',
    'netflix': ',',
    'amusic': ',',
    'yelp': '\t',
    'agames': ','
}