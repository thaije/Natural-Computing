import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Data import
X = pd.read_csv("titanic.csv")

# Preprocessing
y = X["Survived"]
del X["Survived"] # Delete because it is numeric and we want to index independent numerical

X['Sex'].replace(['female','male'], [0,1],inplace=True) # Sex to numerical variable

X.Ticket = X.Ticket.map(lambda x: x[0]) # This transforms Ticket to features of ticket number
X["Ticket"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True) # To numerical values

X["Age"].fillna(X.Age.mean(), inplace=True)

# Training the AdaBoostClassifier
numeric_variables = list(X.dtypes[X.dtypes!= 'object'].index) # Take only numerical variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model.fit(X_train[numeric_variables], y_train)

acc = accuracy_score(y_test, model.predict(X_test[numeric_variables]))
print("Test accuracy:", str(acc))

# Importances
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
features_rank = [numeric_variables[indices[i]] for i in range(len(importances))]

print("Feature ranking:")

for f in range(len(importances)):
    print(str(f+1) +". features:",features_rank[f],"| Importance:", str(importances[indices[f]]))


# Plot the feature importance's of the AdaBoostClassifier
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(importances)), features_rank, rotation='vertical')
plt.xlim([-1, len(importances)])
plt.show()
plt.tight_layout()
