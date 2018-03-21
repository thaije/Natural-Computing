import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


X = pd.read_csv("../data/train.csv")
y = X["Survived"]
del X["Survived"] # Delete because it is numeric and we want to index independent numerical

# Pre processing of some variables
X['Sex'].replace(['female','male'], [0,1],inplace=True) # Sex to numerical variable

X.Ticket = X.Ticket.map(lambda x: x[0]) # This transforms Ticket to features of ticket number
X["Ticket"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True) # To numerical values


X["Age"].fillna(X.Age.mean(), inplace=True)

# #Age to category of age
interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = [0,1,2,3,4,5,6]
X["Age_cat"] = pd.cut(X.Age, interval, labels=cats)
X["Age_cat"].head()

numeric_variables = list(X.dtypes[X.dtypes!= 'object'].index) # Take only numerical variables

###### Model and test: runs to 1 for fast results
runs = 100 # Do it over multiple runs to get estimate of "true" accuracy distribution
accus = np.zeros((runs,1))

for run in range(runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train[numeric_variables], y_train)

    accus[run] = accuracy_score(y_test, model.predict(X_test[numeric_variables]))

print("Mean test accuracy:", str(np.mean(accus)), "| Std:", str(np.std(accus)) + ". in", str(runs),"runs.")
