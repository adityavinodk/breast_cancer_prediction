import numpy as np
import operator
from preProcess import preProcess
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from testResults import testResults

class knn:
    def __init__(self, k, weights):
        # stores the value of k and distance measure as uniform or distance
        self.k = k
        if weights in ['uniform', 'distance']:
            self.weights = weights
    
    def fit(self, x, y):
        # fit x and y as part of training set
        self.x = x
        self.y = y
        self.dataset_size = self.x.shape[0]

    def predict(self, groups):
        # predict for groups of samples
        results = []
        for group in groups:
            results.append(self.predict_single(group))
        return np.array(results)
    
    def predict_single(self, x):
        # predict a single sample

        # Calculate distance measure and square it
        diff = np.tile(x, (self.dataset_size, 1)) - self.x
        sqDiff = diff**2
        distances = (sqDiff.sum(axis=1))**0.5
        
        # sort the distances in ascending order
        sortedDictIndices = distances.argsort()
        
        classCount = {}

        # find the k closest neighbours for each sample
        for i in range(self.k):
            label = self.y[sortedDictIndices[i]]
            # if distance measure is not unform, take weighted average of the nearest neighbours
            if self.weights == 'distance':
                if label not in classCount: classCount[label] = 1
                classCount[label] += 10**15/distances[sortedDictIndices[i]]
            # if distance measure is unform, take mode of the nearest neighbours
            elif self.weights == 'uniform':
                if label not in classCount: classCount[label] = 0
                classCount[label] += 1
        #find the class with highest weight after operation
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
        print("Confusion matrix: \n", confusion_matrix(test_y, pred_y))
        print()
        # print(KNN.predict_single(test_x[5]), test_y[5])