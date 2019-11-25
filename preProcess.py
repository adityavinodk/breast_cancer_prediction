from fileReader import fileReader
import numpy as np
import matplotlib.pyplot as plt

class preProcess:
    def __init__(self):
        self.df_object = fileReader()
        self.df = self.df_object.makeDataframe()
        self.df.drop('code_number', axis=1, inplace=True)
    
    def handle_missing_values(self):
        count = 0
        missing_keys = set()
        for index, row in self.df.iterrows():
            for key in row.keys():
                if row[key] == '?':
                    missing_keys.add(key)
                    count+=1
                    self.df.at[index, key] = np.NAN
        for key in list(missing_keys):
            self.df[key].fillna(self.df[key].median(), inplace=True)
            self.df[key] = self.df[key].astype(int)

        print("Changed ", count, " missing values to -1")
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

    def return_df(self):
        return self.df
    
    def return_correlation_matrix(self):
        return self.df.corr()

    def find_highly_correlated_features(self):
        corr_matrix = self.return_correlation_matrix()
        correlated_features = []
        features = self.df_object.return_features_list()
        features.remove('code_number')
        for i in range(len(corr_matrix)):
            for j in range(0,i):
                if abs(corr_matrix[features[i]][j]) > 0.9: 
                    correlated_features.append((features[i], features[j]))
        return correlated_features
    
    def handle_highly_correlated_features(self):
        dropped = []
        for i in self.find_highly_correlated_features():
            if i[0] not in dropped:
                self.df.drop(i[0], axis=1, inplace=True)
                print(i[0], 'is dropped')
    
if __name__ == '__main__':
    a = preProcess()
    a.handle_missing_values()
    a.data_info()
    a.return_correlation_matrix()
    print(a.return_correlation_matrix())
    print("Highly Correlated features are - ")
    print(a.find_highly_correlated_features())
    a.handle_highly_correlated_features()
    a.data_info()
    # print(a.find_missing_value_columns())
    # a.plot_histogram('bare_nuclei')