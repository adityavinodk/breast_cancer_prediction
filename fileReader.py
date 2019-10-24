import pandas as pd

class fileReader:
    def __init__(self):
        with open('breast-cancer.data', 'r') as openFile:
            with open('data.csv', 'w') as writeCsvFile:
                writeCsvFile.write(openFile.read())
        self.features = ['code_number', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
                         'marginal_adhesion', 'epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitosis']

    def makeDataframe(self):
        features = self.features+['class']
        df = pd.read_csv('data.csv', header=None, names= features)
        return df

    def return_features_list(self):
        return self.features+['class']

if __name__ == '__main__':
    a = fileReader()
    df = a.makeDataframe()
    # print(a.makeDataframe())
    print(a.return_features_list())