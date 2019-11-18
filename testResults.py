class testResults:
    def __init__(self, pred_y, y):
        self.pred_y = pred_y
        self.y = y
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.size = self.y.shape[0]
        for i in range(self.size):
            if self.y[i] == self.pred_y[i]:
                if self.y[i] == 2: self.true_negative += 1
                else: self.true_positive += 1
            elif self.y[i] != self.pred_y[i]:
                if self.pred_y[i] == 2: self.false_negative += 1 
                else: self.false_positive += 1
        
    
    def return_accuracy(self):
        return (self.true_positive+self.true_negative)/self.size
    
    def return_precision(self):
        return self.true_positive/(self.true_positive + self.false_positive)
    
    def retrun_recall(self):
        return self.true_positive/(self.true_positive + self.false_negative)
    
    def return_fscore(self):
        precision = self.true_positive/(self.true_positive + self.false_positive)
        recall = self.true_positive/(self.true_positive + self.false_negative)
        return 2*precision*recall/(precision+recall)