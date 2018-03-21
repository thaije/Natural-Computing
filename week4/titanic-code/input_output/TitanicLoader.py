from input_output.Loader import Loader

# Loader specific for the Titanic task
# The loader loads the data
class TitanicLoader(Loader):

    def load_split(self, training_data_file, test_data_file, verbose=False):


        train, test = self.load_data(training_data_file, test_data_file)
        test_labels = test['PassengerId']
        X_train, Y_train =  self.split_data(train)

        if verbose:
            print( "\n" + ('-' * 40) )
            print( " Original data")
            print( '-' * 40)
            print( X_train.head() )

        print ("Loaded dataset")

        return X_train, Y_train, test, test_labels

    def split_data(self, train):
        # split the features and predector feature
        train_X = train
        train_Y = train_X["Survived"]
        del train_X["Survived"]
        return train_X, train_Y
