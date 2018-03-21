# Base object, which can be used as a Loader for any task
# Preprocesses data such as filling empty data etc.
class Preprocessor(object):
    def preprocess_data(self, data, verbose=False):
        pass

    # handle multiple datasets
    def preprocess_datasets(self, data):
        processedDatasets = []
        for dataset in data:
            processedDatasets.append( self.preprocess_data(dataset) )

        return processedDatasets
