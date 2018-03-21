import pandas as pd
import numpy as np
from preprocessing.Preprocessor import Preprocessor


# Default processor for the titanic task, fills empty fields etc.
class TitanicPreprocessor(Preprocessor):

    # default processing for the Titanic task
    def preprocess_data(self, data, verbose=False):

        print ("Preprocessing data..")

        # replace Sex labels with numerical ID
        # ['female','male'] => [0,1]
        data['Sex'].replace(['female','male'], [0,1], inplace=True)

        # Fill in the mean age for values with missing age
        data["Age"].fillna(data.Age.mean(), inplace=True)

        # Fill in missing Fare attributes with the mean
        data["Fare"].fillna(data.Fare.mean(), inplace=True)

        # Fill in missing embarked attributes with most frequent port
        freq_port = data.Embarked.dropna().mode()[0]
        data['Embarked'] = data['Embarked'].fillna(freq_port)

        # remove the passenger id as it couldn't possibly be predictive
        del data['PassengerId']

        # replace port numbers with numerical ID
        # ['S','C','Q'] => [1,2,3]
        data["Embarked"].replace(['S','C','Q'], [1,2,3], inplace=True)

        if verbose:
            print( "\n" + ('-' * 40) )
            print( "Data After preprocessing")
            print( '-' * 40)
            print( data.head() )

        return data
