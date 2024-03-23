from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, y_prob=None):
        """
        Initialize the metrics class with true labels, predictions, and probabilities.

        Parameters:
        - y_true: True class labels.
        - y_pred: Predicted class labels.
        - y_prob: Probabilities for the positive class. Required for ROC-AUC.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='binary')

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='binary')

    def f1(self):
        return f1_score(self.y_true, self.y_pred, average='binary')

    def roc_auc(self):
        if self.y_prob is None:
            raise ValueError("y_prob is required for ROC-AUC calculation.")
        return roc_auc_score(self.y_true, self.y_prob)

    def get_all_metrics(self):
        """
        Returns all metrics in a dictionary.
        """
        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1 Score': self.f1()
        }
        if self.y_prob is not None:
            metrics['ROC-AUC'] = self.roc_auc()
        return metrics
