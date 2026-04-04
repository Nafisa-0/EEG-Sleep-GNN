from sklearn.metrics import f1_score, confusion_matrix
import logging

def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def compute_metrics(y_true, y_pred):
    acc = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    return acc, f1, cm