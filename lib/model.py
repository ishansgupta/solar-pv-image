from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

class Model:

    def __init__(self, x, y):
        self.X_train = x
        self.y_train = y

    def gradient_boosting(self, **params):
        self.reshape()
        clf = GradientBoostingClassifier(**params)
        return clf.fit(self.X_train, self.y_train)

    def svm_classification(self, **params):
        self.reshape()
        clf = svm.SVC(**params)
        return clf.fit(self.X_train, self.y_train)

    def reshape(self):
        self.X_train = self.X_train.reshape(len(self.X_train),-1)