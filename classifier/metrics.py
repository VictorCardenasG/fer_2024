# Source code module metrics calculations

from sklearn.metrics import accuracy_score

def calculate_metric(y, y_pred):
    return accuracy_score(y, y_pred)
