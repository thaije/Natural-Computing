import numpy as np


def monteCarloCountingOnes(maxScore, n, maxIter):
    arr = np.random.randint(2, size=(n,))
    index = 0

    bestArr = arr

    # try to get the max score within x iterations
    while np.sum(arr) < maxScore and index < maxIter:
        # create a random bitstring
        arr = np.random.randint(2, size=(n,))
        index += 1

        # save best bitstring
        if np.sum(arr) > np.sum(bestArr):
            bestArr = arr

    return bestArr, np.sum(arr)


n = 100
maxScore = n
maxIter = 1500
bestIndividual, bestF = monteCarloCountingOnes(maxScore, n, maxIter)

print "Best individual"
print "     Bitstring:", bestIndividual
print "     Fitness:", bestF , " ( max is", maxScore, ")"

# TODO: Plot shit
