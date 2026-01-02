from abc import ABC, abstractmethod

class Probe(ABC):
    def __init__(self, cfg):
        """
        Initialize the probe.
        Args:
            cfg (ConfigDict): ConfigDict with probe parameters.
        """
        self.cfg = cfg
        
    @abstractmethod
    def fit(self, train_dataset, validation_dataset):
        """
        Fits the probe to training data.
        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, dim], train_dataset['y'] has shape [batch_size].
            validation_dataset (dict): validation_dataset['X'] has shape [batch_size, dim], validation_dataset['y'] has shape [batch_size].
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Get prediction labels (0 or 1) for the dataset.
        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        Returns:
            y_pred (tensor): predicted labels of shape [batch_size].
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Get prediction probabilities of each point being class 1 for the dataset.
        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        Returns:
            y_pred_proba (tensor): predicted probabilities of shape [batch_size].
        """
        pass

    @abstractmethod
    def eval(self, test_dataset):
        """
        Evaluates the probe on the test dataset.
        Args:
            test_dataset (dict): test_dataset['X'] has shape [batch_size, dim], test_dataset['y'] has shape [batch_size].
        Returns:
            results (dict): dictionary with the following keys:
                - accuracy (float): accuracy score.
                - roc_auc (float): roc auc score.
                - tpr_at_1_fpr (float): tpr at 1% fpr.
        """
        pass