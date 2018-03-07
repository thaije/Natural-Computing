import math
from copy import deepcopy
import numpy as np

# read in an datafile and remove the label column
def readDataset(filename):
    datafile = open(filename, "r")
    x, y = [], []
    for line in datafile:
        lin = line.strip().split() # split on spaces too for other dataset
        x.append(map(float, lin[:-1]))
        y.append(lin[-1])
    return x, y

# Calculate the eucladian distance from datavector z to centroid m
def eucladianDistance(z, m):
    return np.sqrt(np.sum(np.power(np.array(z) - np.array(m), 2)))


class Particle(object):
    """ PSO particle """

    def __init__(self, nClusters, x):
        self.classifiedDatapoints = []
        self.datapointsDistToCentroids = np.zeros((nClusters, len(x)))
        # init velocity with 0
        self.vel = np.zeros( (nClusters, len(x[0])) )
        # pos corresponds to possible locations of centroids
        self.pos = []
        self.initPosition(nClusters, x)
        self.bestPos = deepcopy(self.pos)
        self.fit = self.updateFitness(nClusters, x)
        self.bestFit = deepcopy(self.fit)

    # The position of a particle are possible positions of the centroids
    def initPosition(self, nClusters, x):
        # add a coordinate for each cluster
        for i in range (0, nClusters):
            coord = x[np.random.randint(0, len(x) - 1)]
            # A coordinate is a random datapoint from the dataset, where the
            # cluster coordinates of a particle cannot be the same
            while coord in self.pos:
                coord = x[np.random.randint(0, len(x)-1)]
            self.pos.append(coord)

    # update fitness of current particle
    def updateFitness(self, nClusters, x):
        fit_i = 0
        for cluster_i in range(nClusters):
            cluster_data = self.datapointsDistToCentroids[cluster_i][self.classifiedDatapoints == cluster_i]
            if len(cluster_data) != 0:
                cluster_fit = np.sum(cluster_data) / len(cluster_data)
                fit_i += cluster_fit
        self.fit = fit_i / nClusters

    def updatePersonalBest(self):
        if self.fit < self.bestFit or self.bestFit is None:
            self.bestPos = deepcopy(self.pos)
            self.bestFit = deepcopy(self.fit)

    def updateVelocity(self, socialBest, inertia, acc_c1, acc_c2, x):
        rho1 = np.random.random_sample((1, len(x[0])))
        rho2 = np.random.random_sample((1, len(x[0])))
        self.vel = [map(float,j) for j in inertia * np.array(self.vel) + acc_c1 * rho1 * np.array(np.array(self.bestPos) - np.array(self.pos)) + acc_c2 * rho2 * np.array(np.array(socialBest.pos) - np.array(self.pos))]

    def updatePosition(self):
        self.pos = [map(float,j) for j in np.array(self.pos) + self.vel]

    # calc the distance of each datapoint to each centroid
    def classifyDatapoints(self, x, nClusters):
        for cluster_i in range(nClusters):
            for datapoint in range(len(x)):
                self.datapointsDistToCentroids[cluster_i][datapoint] = eucladianDistance(x[datapoint], self.pos[cluster_i])
        self.classifiedDatapoints = np.argmin(self.datapointsDistToCentroids, axis=0)


class Swarm(object):
    """ Swarm containing PSO particles """

    def __init__(self, nParticles, nClusters, x):
        # init particles
        self.particles = [Particle(nClusters, x) for i in range (0, nParticles)]
        self.socialBest = None
        self.updateSocialBest()
        self.nClusters = nClusters

    def updateSocialBest(self):
        for p in self.particles:
            if self.socialBest is None or self.socialBest.fit is None or p.fit < self.socialBest.fit:
                self.socialBest = deepcopy(p)

    def updateFitnesses(self, x):
        for p in self.particles:
            p.updateFitness(self.nClusters, x)
            p.updatePersonalBest()

    def classifyDatapoints(self,x):
        for p in self.particles:
            p.classifyDatapoints(x, self.nClusters)

    def updatePositions(self, inertia, acc_c1, acc_c2, x):
        for p in self.particles:
            p.updateVelocity(self.socialBest, inertia, acc_c1, acc_c2, x)
            p.updatePosition()



def PSO(nClusters, nParticles, inertia, acc_c1, acc_c2, maxIter, initPersonalBest, initSocialbest, dataFile):
    x, y = readDataset(dataFile)
    swarm = Swarm(nParticles, nClusters, x)
    print ("Initialized swarm")

    # run for maxIter iterations and update the particles to find a solution
    for t in xrange(0, maxIter):
        swarm.classifyDatapoints(x)

        # update the fitness of each particle, and check for new  personal best
        swarm.updateFitnesses(x)

        # update new social best (best of all particles)
        swarm.updateSocialBest()
        print "Iteration %s with best particle fitness %s" % (t, swarm.socialBest.fit)

        # update velocities and positions of the particles
        swarm.updatePositions(inertia, acc_c1, acc_c2, x)

    # update fitnesses and social best last time
    swarm.classifyDatapoints(x)
    swarm.updateFitnesses(x)
    swarm.updateSocialBest()

    print ("Best particle has fitness %s" % swarm.socialBest.fit)
    print ("Best particle position:")
    print (swarm.socialBest.pos)



if __name__ == '__main__':
    # constants
    nClusters = 2
    nParticles = 30
    inertia_w = 0.72
    acceleration_c1= 1.49
    acceleration_c2 = 1.49
    maxIter = 30
    initPersonalBest = 1.5
    initSocialbest = 1.5
    dataFile = "data/artificial.data"

    PSO(nClusters, nParticles, inertia_w, acceleration_c1, acceleration_c2, maxIter, initPersonalBest, initSocialbest, dataFile)
