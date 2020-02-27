from sklearn import metrics

class Evaluator():
    def __init__(self, y_actual, y_hat, y_hat_proba):
        self.y_actual = y_actual
        self.y_hat = y_hat
        self.y_hat_proba = y_hat_proba

    def measures(self):
        auc_score = metrics.roc_auc_score(self.y_actual, self.y_hat_proba[:, 1])
        accuracy = metrics.accuracy_score(self.y_actual, self.y_hat)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_actual, self.y_hat_proba[:, 1])
        return {"measures":{"area_under_curve":auc_score, "accuracy": accuracy}, "roc":{"fpr": fpr, "tpr": tpr,
         "thresholds": thresholds}}
