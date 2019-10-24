from fileReader import fileReader
import numpy as np
import matplotlib.pyplot as plt

class preProcess:
    def __init__(self):
        self.df_object = fileReader()
        self.df = self.df_object.makeDataframe()
    
    def handle_missing_values(self):
        count = 0
        for index, row in self.df.iterrows():
            for key in row.keys():
                if row[key] == '?':
                    count+=1
                    self.df.at[index, key] = np.NAN
        print("Changed ", count," missing values to -1")
        print("----------------------------------------------")

    def find_missing_value_columns(self):
        print("List of missing values - ")
        return [column for column in self.df.columns.values if self.df[column].isnull().sum()>0]

    def data_info(self):
        print(self.df.info())
        print(self.df.describe())
        print("----------------------------------------------")

    def plot_histogram(self, key):
        print(self.df[key].value_counts())
        plt.hist(self.df[key])
        plt.show()
    
    # def return_correlation_matrix(self):
    #     return self.df.corr()

    # def find_highly_correlated_features(self):
    #     corr_matrix = self.return_correlation_matrix()
    #     correlated_features = []
    #     features = self.df_object.return_features_list()
    #     for i in range(len(corr_matrix)):
    #         for j in range(0,i):
    #             if abs(corr_matrix[features[i]][j]) > 0.9: 
    #                 correlated_features.append((features[i], features[j]))
    #     return correlated_features
    
if __name__ == '__main__':
    a = preProcess()
    a.handle_missing_values()
    a.data_info()   
    # print(a.return_correlation_matrix())
    # print(a.find_missing_value_columns())
    # a.plot_histogram('bare_nuclei')