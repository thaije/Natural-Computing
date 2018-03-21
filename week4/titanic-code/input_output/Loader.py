import pandas as pd

# Base object, which can be used as a Loader for any task
# The loader loads the data
class Loader(object):
    def load_split(self, training_data_file, test_data_file, verbose=False):
        pass

    def load_data(self, training_data_file, test_data_file = None):
        train = pd.read_csv(training_data_file)
        test = pd.read_csv(test_data_file)
        return train, test

    def split_data(self, train):
        pass
