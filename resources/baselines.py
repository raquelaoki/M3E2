
class methodNAME:
    def __init__(self,X_train, X_test, y_train, y_test):
        super(methodNAME, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print('Running methodNAME')

    def fit(self, parameters):
        print('adding fitting part')
        print('should output predictions and some metrics for evaluation')
        self.model = model
        y_test_pred = pred(X_test) # generic pred function, depends on the method
        return f1_score(self.y_test, y_test_pred)

    def cate(self):
        print('optitional - depends on the methods implementation')
        # one value per treatment 
        return estimated_cate


"""
Usage:
Check the notebook with examples of the other baselines

1. Make sure they all receive the same split X_train, X_test, y_train, y_test
2. Return CATE, return f1_score for testing set
"""
