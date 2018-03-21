# Base object, which can be used as a Loader for any task
# The Features object does the feature engineering
class Features(object):
    def engineer_features(self, data, verbose=False):
        pass

    def engineer_features_multiple_ds(self, data):
        processedDatasets = []
        for dataset in data:
            processedDatasets.append( self.engineer_features(dataset) )
        return processedDatasets
