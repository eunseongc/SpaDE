import os

class BaseDataset:
    def __init__(self, data_dir, data_name, triple_name, doc_id, test_set, expand_collection):
        """
        Dataset class

        :param str data_dir: base directory of data
        :param str dataset: Name of dataset 
        """
        self.data_dir = data_dir
        self.data_name = data_name
        self.triple_name = triple_name
        self.doc_id = doc_id
        self.test_set = test_set
        self.expand_collection = expand_collection

    def check_dataset_exists(self):
        return os.path.exists(self.docs_data_file)

    def __str__(self):
        return 'BaseDataset'
