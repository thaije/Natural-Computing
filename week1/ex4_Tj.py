
import numpy as np
import random
import copy


def decision(probability):
    return random.random() < probability


def OnePlusOneGA(maxScore, n, maxIter, mutation):
    arr = np.random.randint(2, size=(n,))
    index = 0
    bestArr = arr

    # try to get the max score within x iterations
    while np.sum(arr) < maxScore and index < maxIter:
        index += 1
        oldArr = copy.copy(arr)

        # loop through the bitstring
        for x in range(0, len(arr)):
            # randomly mutate/invert items based on p
            if decision(mutation):
                arr[x] = 1 if arr[x] == 0 else 0

        # only continue with the new one if it is better
        if np.sum(arr) < np.sum(oldArr):
            arr = oldArr

        # save best bitstring
        if np.sum(arr) > np.sum(bestArr):
            bestArr = arr

    return bestArr, np.sum(arr)

n = 100
maxScore = n
maxIter = 1500
mutation = 1.0 / n
bestIndividual, bestF = OnePlusOneGA(maxScore, n, maxIter, mutation)

print "Best individual"
print "     Bitstring:", bestIndividual
print "     Fitness:", bestF , " ( max is", maxScore, ")"

# TODO: Plot shit
