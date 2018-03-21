import pandas as pd
import numpy as np


def testAccuracies(modelNames):
    return [testAccuracy(model) for model in modelNames]


def testAccuracy(modelName):
    model = pd.read_csv('predictions/' + modelName + '.csv')
    model = model.drop('PassengerId', axis=1)
    model = np.array(model)

    test_file_path = "Data/fullData/test_complete.csv"
    test_file = pd.read_csv(test_file_path)
    test_file = np.array(test_file['Survived'])

    overlap = 0.0
    for i in range(len(model)):
        if model[i] == test_file[i]:
            overlap += 1.0

    acc = overlap / len(model)
    # print(modelName , " model has accuracy " , acc)
    return acc



# testAccuracies(["Bayes", "GRBT", "KNN", "MLP", "RF", "SVM", "XGBoost", "VotingEnsemble"])
