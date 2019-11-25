from preProcess import preProcess
from sklearn.model_selection import KFold
from collections import Counter
from testResults import testResults
from decisionTree import decisionTree
import numpy as np
import operator

class randomForest:
    def __init__(self, criterion='entropy', n_trees=10 ,max_depth = None, min_samples_split=2, n_features='log2'):
        np.random.seed(49)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.n_trees = n_trees
        self.trees = [None] * self.n_trees
        
    def fit(self, dataset):
        if self.n_features=='log2':
            self.n_features = int(np.log2(dataset.shape[1]-1))
        elif self.n_features=='sqrt':
            self.n_features = int(np.sqrt(dataset.shape[1]-1))
        for i in range(self.n_trees):
            self.trees[i] = self.buildTree(dataset)
        
    def buildTree(self, dataset):
        dt = decisionTree(self.criterion, self.max_depth, self.min_samples_split)
        random_locations = np.random.permutation(len(dataset))
        random_features = np.random.permutation(len(dataset[0])-1)[:self.n_features]
        new_dataset = []
        for i in range(len(dataset)):
            if i in random_locations:
                sample = dataset[i]
                newSample = []
                for j in range(0, len(dataset[0])-1):
                    if j in random_features:
                        newSample.append(sample[j])
                newSample.append(sample[-1])
            new_dataset.append(newSample)
        dt.fit(np.array(new_dataset))
        return dt
    
    def predict(self, groups):
        results = []
        for group in groups:
            results.append(self.predict_single(group))
        return np.array(results)
    
    def predict_single(self, x):
        labelCounts = {2:0, 4:0}
        for dt in self.trees:
            labelCounts[dt.predict_single(x)]+=1
        return max(labelCounts.items(), key=operator.itemgetter(1))[0]

#change n_trees parameter and n_features('log2' or 'sqrt') to find optimal accuracy
if __name__ == '__main__':
    preProcessData = preProcess()
    preProcessData.handle_missing_values()
    preProcessData.handle_highly_correlated_features()
    df = preProcessData.return_df()
    RF = randomForest(n_features='log2', n_trees=30)
    kFold = KFold(6, True, 1)
    values = df.values
    for train, test in kFold.split(values):
        print("Taking %d train datapoints" % len(train))
        train_x = values[train]
        test_x, test_y = values[test][:,:-1], values[test][:,-1]
        RF.fit(train_x)
        pred_y = RF.predict(test_x)
        results = testResults(pred_y, test_y)
        print("Accuracy of model is ", results.return_accuracy())
        print('F Score of model is ', results.return_fscore())
        print()