from preProcess import preProcess
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from collections import Counter
from testResults import testResults
from decisionTree import decisionTree
import numpy as np
import operator

class randomForest:
    def __init__(self, criterion='entropy', n_trees=10 ,max_depth = None, min_samples_split=2, n_features='full', sample_size=300):
        # consists of a number of parameters
        # n_features is the criteria on which number of features for a particular random decision tree is decided
        # n_trees is the number of trees used for the bagging process
        np.random.seed(49)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = [None] * self.n_trees
        
    def fit(self, dataset):
        # fit the dataset onto the model
        if self.n_features=='log2':
            self.n_features = int(np.log2(dataset.shape[1]-1))
        elif self.n_features=='sqrt':
            self.n_features = int(np.sqrt(dataset.shape[1]-1))
        elif self.n_features=='full':
            self.n_features = dataset.shape[1]-1
        # we create n_trees number of random decision trees for the dataset
        for i in range(self.n_trees):
            self.trees[i] = self.buildTree(dataset)
        
    def buildTree(self, dataset):
        # create a decision tree
        dt = decisionTree(self.criterion, self.max_depth, self.min_samples_split)
        # randomly choose n samples 
        random_locations = np.random.permutation(len(dataset))[:self.sample_size]
        # randomly choose features 
        random_features = np.random.permutation(len(dataset[0])-1)[:self.n_features]
        new_dataset = []
        # create a dataset with the following constraints
        for i in range(len(dataset)):
            if i in random_locations:
                sample = dataset[i]
                newSample = []
                for j in range(0, len(dataset[0])-1):
                    if j in random_features:
                        newSample.append(sample[j])
                newSample.append(sample[-1])
                new_dataset.append(newSample)
        # fit the dataset onto the decision tree
        dt.fit(np.array(new_dataset))
        return dt
    
    def predict(self, groups):
        #predict for group of samples
        results = []
        for group in groups:
            results.append(self.predict_single(group))
        return np.array(results)
    
    def predict_single(self, x):
        # predict for single sample
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
    RF = randomForest(n_features='log2', n_trees=30, sample_size=400)
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
        print("Confusion matrix:\n", confusion_matrix(test_y, pred_y))
        print()