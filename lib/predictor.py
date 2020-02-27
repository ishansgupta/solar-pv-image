from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

class Predictor:

    def __init__(self, fit, X_test, reshape = False):
        self.fit = fit
        self.X_test = X_test
        if reshape:
            self.reshape()

    def predict_proba(self):
        return self.fit.predict_proba(self.X_test)
        
    def predict(self):
        return self.fit.predict(self.X_test)

    def reshape(self):
        self.X_test = self.X_test.reshape(len(self.X_test),-1)