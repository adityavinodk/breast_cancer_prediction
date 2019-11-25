from math import log
from preProcess import preProcess
from sklearn.model_selection import KFold
from collections import Counter
from testResults import testResults
import numpy as np

class node:
    def __init__(self, attribute=None, depth=None):
        self.depth = depth
        self.children = [None] * 10
        self.attribute = attribute
        self.definiteLabel = None

    def assign_child(self, attribute_value, child):
        self.children[attribute_value] = child

    def assign_label(self, label):
        self.definiteLabel = label

class decisionTree:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_built = False

    def find_entropy(self, dataset):
        numEntries = len(dataset)
        labelCounts = Counter()
        for i in range(numEntries):
            labelCounts[dataset[i][-1]] += 1
        shannonEntropy = 0
        for label in labelCounts:
            prob = float(labelCounts[label])/numEntries
            shannonEntropy -= prob*log(prob, 2)
        return shannonEntropy

    def perform_split(self, dataset, axis, value):
        newDataset = []
        listDataset = list(dataset)
        for sample in listDataset:
            if sample[axis] == value:
                reducedFeatureVector = list(sample[:axis])
                reducedFeatureVector.extend(list(sample[axis+1:]))
                newDataset.append(reducedFeatureVector)
        return np.array(newDataset)

    def chooseBestFeatureToSplit(self, dataset):
        numFeatures = len(dataset[0])-1
        baseEntropy = self.find_entropy(dataset)
        bestInfoGain = 0
        bestFeature = -1
        for i in range(numFeatures):
            entropyVal = 0
            for value in range(1, 11):
                subDataset = self.perform_split(dataset, i, value)
                prob = len(subDataset) / float(len(dataset))
                entropyVal += prob*self.find_entropy(subDataset)
            infoGain = baseEntropy - entropyVal
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def returnClassProbabilities(self, dataset):
        sampleCount = len(dataset)
        labelCounts = Counter()
        for i in range(sampleCount):
            if dataset[i][-1] == 2:
                labelCounts[2] += 1
            else:
                labelCounts[4] += 1
        return [labelCounts[2]/float(sampleCount), labelCounts[4]/float(sampleCount)]

    def recursiveBuild(self, dataNode, dataset):
        probs = self.returnClassProbabilities(dataset)
        probs = sorted(self.returnClassProbabilities(dataset), reverse=True)
        if probs[0] > 90 or dataNode.depth == self.max_depth or len(dataset[0])<self.min_samples_split:
            dataNode.assign_label(probs[0])
            return

        dataNode.labelSamples = [2,4][np.argmax(probs)]
        for i in range(1, 11):
            splitDataset = self.perform_split(dataset, dataNode.attribute, i)
            if len(splitDataset):
                best_feature = self.chooseBestFeatureToSplit(splitDataset)
                if best_feature==-1:
                    subProbs = self.returnClassProbabilities(splitDataset)
                    dataNode.children[i-1] = node(depth = dataNode.depth+1)
                    dataNode.children[i-1].assign_label([2,4][np.argmax(subProbs)])
                else:    
                    dataNode.children[i-1] = node(attribute=best_feature, depth = dataNode.depth+1)
                    self.recursiveBuild(dataNode.children[i-1], splitDataset)

    def fit(self, dataset):
        self.tree_built = True
        self.root = node(attribute = self.chooseBestFeatureToSplit(dataset), depth=0)
        self.recursiveBuild(self.root, dataset)
    
    def predict(self, groups):
        results = []
        for group in groups:
            results.append(self.predict_single(group))
        return np.array(results)
    
    def predict_single(self, x):
        available_features = list(x)
        # print(available_features)
        if self.tree_built:
            pres_node = self.root
            while pres_node.definiteLabel == None:
                attribute = pres_node.attribute
                if pres_node.children[available_features[attribute]-1]:
                    pres_node = pres_node.children[available_features[attribute]-1]
                    available_features.pop(attribute)
                else: return pres_node.labelSamples
                # print(available_features)
        return pres_node.definiteLabel
            
        
if __name__ == '__main__':
    preProcessData = preProcess()
    preProcessData.handle_missing_values()
    preProcessData.handle_highly_correlated_features()
    df = preProcessData.return_df()
    DT = decisionTree()
    kFold = KFold(6, True, 1)
    values = df.values
    for train, test in kFold.split(values):
        print("Taking %d train datapoints" % len(train))
        train_x = values[train]
        test_x, test_y = values[test][:,:-1], values[test][:,-1]
        DT.fit(train_x)
        pred_y = DT.predict(test_x)
        results = testResults(pred_y, test_y)
        print("Accuracy of model is ", results.return_accuracy())
        print('F Score of model is ', results.return_fscore())
        print()
        # print(DT.predict_single(test_x[0]), test_y[0])