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


n_estimators = [50, 100, 150, 200]
learning_rates = np.arange(0.1, 1, .2)

bestScore = 0
bestModel = 0

for learn_rate in learning_rates:

    x = []
    y = []
    for n_est in n_estimators:
        model = AdaBoostClassifier(n_estimators=n_est, learning_rate=learn_rate)
        model.fit(X_train[numeric_variables], y_train)
        acc = accuracy_score(y_test, model.predict(X_test[numeric_variables]))
        print("Test accuracy %0.2f for n_est %d and learn_rate %0.2f" % (acc, n_est, learn_rate))

        if acc > bestScore:
            bestScore = acc
            bestModel = model

        x.append(n_est)
        y.append(acc)

    plt.plot(x, y, Label="learning_rate = " + str(round(learn_rate, 2)))


# Importances
importances = bestModel.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
features_rank = [numeric_variables[indices[i]] for i in range(len(importances))]
print("Feature ranking of best model:")
for f in range(len(importances)):
    print(str(f+1) +". features:",features_rank[f],"| Importance:", str(importances[indices[f]]))


# plot the accuracies of the adaboost with different parameters
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.legend()
plt.show()


# Plot the feature importance's of the AdaBoostClassifier
plt.figure()
plt.title("Feature importances for best model")
plt.bar(range(len(importances)), importances[indices],
       color="r", align="center")
plt.xticks(range(len(importances)), features_rank, rotation='vertical')
plt.xlim([-1, len(importances)])
plt.show()
plt.tight_layout()
