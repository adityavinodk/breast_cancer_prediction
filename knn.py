import numpy as np
import operator
from preProcess import preProcess
from sklearn.model_selection import KFold
from testResults import testResults

class knn:
    def __init__(self, k, weights):
        self.k = k
        if weights in ['uniform', 'distance']:
            self.weights = weights
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.dataset_size = self.x.shape[0]

    def predict(self, groups):
        results = []
        for group in groups:
            results.append(self.predict_single(group))
        return np.array(results)
    
    def predict_single(self, x):
        diff = np.tile(x, (self.dataset_size, 1)) - self.x
        sqDiff = diff**2
        distances = (sqDiff.sum(axis=1))**0.5
        sortedDictIndices = distances.argsort()
        # print(max(distances))
        classCount = {}
        for i in range(self.k):
            label = self.y[sortedDictIndices[i]]
            if self.weights == 'distance':
                if label not in classCount: classCount[label] = 1
                classCount[label] += 10**15/distances[sortedDictIndices[i]]
            elif self.weights == 'uniform':
                if label not in classCount: classCount[label] = 0
                classCount[label] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        predVal = sortedClassCount[0][0]
        return predVal

if __name__ == '__main__':
    preProcessData = preProcess()
    preProcessData.handle_missing_values()
    preProcessData.handle_highly_correlated_features()
    df = preProcessData.return_df()
    KNN = knn(5, 'uniform')
    kFold = KFold(6, True, 1)
    values = df.values
    for train, test in kFold.split(values):
        print("Taking %d train datapoints" % len(train))
        train_x, train_y = values[train][:,:-1], values[train][:,-1]
        test_x, test_y = values[test][:,:-1], values[test][:,-1]
        KNN.fit(train_x, train_y)
        pred_y = KNN.predict(test_x)
        results = testResults(pred_y, test_y)
        print("Accuracy of model is ", results.return_accuracy())
        print('F Score of model is ', results.return_fscore())
        print()
        # print(KNN.predict_single(test_x[5]), test_y[5])