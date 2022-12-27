import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, dataset, model_conf):
        super(BaseModel, self).__init__()
        """
        Initialize model configuration.

        Parameters:

        :param Dataset dataset: dataset to use
        :param Parameters model_conf: model configurations such as hidden dimension

        """
        pass

    def forward(self, *input):
        """
        Pytorch forward path.
        return output

        """

        raise NotImplementedError

    def train_model(self, dataset, evaluator, early_stop, saver, logger, config):
        """
        Train model following given config.

        """
        raise NotImplementedError

    def predict(self, dataset):
        """
        Make prediction on eval data which is stored in dataset.
        evaluation data is stored at dataset.eval_input as matrix form.

        :param Dataset dataset: dataset to use

        :returns eval_output: (num_users, num_items) shaped matrix with predicted scores
        """
        raise NotImplementedError