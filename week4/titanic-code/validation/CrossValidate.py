from sklearn import model_selection

# Wrapper methods for various cross validation methods of Scikit learn

# Finds the optimal Classifier Fit for the given train_test_split.
def cv_scores(X, y, clf, train_test_split):
    scores = []
    for train, test in train_test_split:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
    return scores

def KFold(X, y, clf, K=10):
    print ("KFold validation with K=%d" % K)
    kf = model_selection.KFold(n_splits=K)
    return cv_scores(X, y, clf, kf.split(X))

def RepeatedKFold(X, y, clf, K=10, N=10):
    print ("RepeatedKFold validation with K=%d and N=%d" % (K, N))
    rkf = model_selection.RepeatedKFold(n_splits=K, n_repeats=N)
    return cv_scores(X, y, clf, rkf.split(X))

def StratifiedKFold(X, y, clf, K=10):
    print ("StratifiedKFold validation with K=%d" % K)
    skf = model_selection.StratifiedKFold(K)
    return cv_scores(X, y, clf, skf.split(X, y))

def LeaveOneOut(X, y, clf):
    print ("LeaveOneOut validation")
    loo = model_selection.LeaveOneOut()
    return cv_scores(X, y, clf, loo.split(X))
