from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def readtext(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def preprocess(content):
    for i in range(len(content)):
        content[i] = content[i].split(",")
        content[i] = content[i][:-1] # Remove labels
        for j in range(len(content[i])):
            content[i][j] = float(content[i][j])
    content = content[:-1]
    content = np.asarray(content, dtype=np.float32)
    return content

def d_L2(z, m):
    """
    :param z: data vector p
    :param m: centroid j
    :return: distance of vector to centroid
    """
    d = np.sqrt(np.sum(np.power(z-m,2)))
    return d

def fitness_Kmeans(kmeans, data, Nc, distFunc):
    """
    :param kmeans: kmeans variable that is the output of sklearn.cluster
    :param data: data
    :param Nc: number of clusters
    :param distFunc: euclidean distance function
    :return: fitness of the best kmeans centroids
    """
    Ndata = np.shape(data)[0]

    distances = np.zeros((Nc, Ndata))

    for cluster_i in range(Nc):
        for datapoint in range(Ndata):
            distances[cluster_i][datapoint] = distFunc(data[datapoint], kmeans.cluster_centers_[cluster_i])

    fit_i = 0
    for cluster_i in range(Nc):
        cluster_data = distances[cluster_i][kmeans.labels_ == cluster_i]
        cluster_fit = np.sum(cluster_data) / len(cluster_data)
        fit_i += cluster_fit

    fitness = fit_i /Nc
    return fitness

class Particle:
    def __init__(self, Nd, Nc, bounds):
        self.minbound = bounds[0]
        self.maxbound = bounds[1]
        self.centroids = np.random.uniform(low=self.minbound, high=self.maxbound, size=(Nc,Nd)) # Centroids is position of all centroids
        self.velocity = np.random.uniform(low=self.minbound, high=self.maxbound, size=(Nc,Nd)) # particle velocity per centroid
        self.pos_best=[]           # best position individual
        self.fitness_best=-1          # best error individual
        self.fitness=-1               # error individual
        self.distances = []            # Distance in L2 norm (equation 1 paper)
        self.assignments=[]            # Cluster number that data point i currently belongs to
        self.Nc = Nc                # Number of clusters
        self.Nd = Nd                # Number of dimensions in data


    def assign(self, data, distFunc):
        """
        :param data: data to be clustered
        :param distFunc: function that calculates euclidean distances
        :return: an assignment for every datapoint to cluster j
        """
        Ndata = np.shape(data)[0]
        self.distances = np.zeros((self.Nc, Ndata))

        for cluster_i in range(self.Nc):
            for datapoint in range(Ndata):
                self.distances[cluster_i][datapoint] = distFunc(data[datapoint], self.centroids[cluster_i])

        self.assignments = np.argmin(self.distances, axis=0)


    def update_fitness(self):
        """
        :return: new fitness of the particle
        """
        fit_i = 0
        for cluster_i in range(self.Nc):
            cluster_data = self.distances[cluster_i][self.assignments == cluster_i]
            if len(cluster_data) == 0: pass
            else:
                cluster_fit = np.sum(cluster_data) / len(cluster_data)
                fit_i += cluster_fit

        self.fitness = fit_i / self.Nc

        if self.fitness < self.fitness_best or self.fitness_best == -1:
            self.fitness_best = np.copy(self.fitness)
            self.pos_best = np.copy(self.centroids)

    def update_velocity(self, pos_best_g):
        """
        :param pos_best_g: Position of the current best particle centroids
        :return: Updated position of particle in the cluster and data dimensions
        """
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognitive constant
        c2 = 2  # social constant

        r1 = np.random.uniform(low=0, high=1, size=(self.Nc, self.Nd)) # For stochastic
        r2 = np.random.uniform(low=0, high=1, size=(self.Nc, self.Nd))

        vel_cognitive = c1 * r1*  (self.pos_best - self.centroids)
        vel_social = c2 * r2 * (pos_best_g - self.centroids)
        self.velocity = w * self.velocity + vel_cognitive + vel_social


    def update_position(self):
        self.centroids = self.centroids + self.velocity

        # Bounds of the centroids - since data is ~U(min, max)
        self.centroids[self.centroids > self.maxbound] = self.maxbound
        self.centroids[self.centroids < self.minbound] = self.minbound

def PSO(data, Nc, Nd, bounds, n_particles=10, n_iter=30, HYBRID=0):
    pos_best_g = []
    fitness_best_g = -1
    fitness_besties = []

    # Initialize swarm
    swarm = []
    for i in range(n_particles):
        swarm.append(Particle(Nd, Nc, bounds))

    if HYBRID == 1:  # If hybrid model is on
        kmeans = KMeans(n_clusters=Nc, random_state=0, max_iter=n_iter, tol=0.0001).fit(data)
        swarm[0].centroids = kmeans.cluster_centers_

    # PSO loop
    for iter in range(n_iter):

        for j in range(n_particles):
            swarm[j].assign(data, d_L2)
            swarm[j].update_fitness()

            if swarm[
                j].fitness < fitness_best_g or fitness_best_g == -1:  # Check if current particle is the new global best
                fitness_best_g = np.copy(swarm[j].fitness)
                pos_best_g = np.copy(swarm[j].centroids)
        fitness_besties.append(fitness_best_g)

        for j in range(n_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position()

    return fitness_best_g, fitness_besties

# General parameters
n_iter = 30 # Paper uses 1000?
n_particles = 10 # Number of particles as in paper
Nc = range(2,10)

PROBLEM = 1 # 0 for Iris data and 1 for Artificial problem

if PROBLEM == 1: # if problem is artificial data 1
    # Artificial problem 1
    Nd = 2 # Number of dimensions
    data = np.random.uniform(low=-1, high=1, size=(400, Nd)) # Random data

    ############################################
    # # add class generated from formula as extra dimension
    # Nd += 1
    # newD = np.zeros((len(data), Nd))
    # for i in range(len(data)):
    #     val = 0.0
    #     # execute formula to determine if value is 1 or 0
    #     if (data[i][0] >= 0.7 or data[i][0] <= 0.3) and (data[i][1] >= -0.2 - data[i][0]):
    #         val = 1.0
    #     newD[i] = np.insert(data[i], 2, val)
    # data = newD

    bounds = [-1, 1]
    Nc = range(2,10) # Try clusters 2 to 10 - like paper
    n_simulations = 1 # More takes too long
    besties = np.zeros((3, len(Nc), n_simulations))

if PROBLEM == 0: # If problem is Iris data
    # Iris data
    data = readtext("iris.data")
    data = preprocess(data)
    bounds = [np.min(data), np.max(data)]
    Nd = np.shape(data)[1]
    Nc = range(3,4) # This means 3: just so I can iterate
    n_simulations = 10
    besties = np.zeros((3,len(Nc), n_simulations))

# START Comparing methods loop
for sim_i in range(n_simulations):
    for Nc_i in Nc:
        print("Simulation:",str(sim_i + 1),"|", "Cluster:", str(Nc_i))
        [best_fit_PSO, d] =PSO(data=data, Nc=Nc_i, Nd=Nd,bounds=bounds, n_particles=n_particles, n_iter=n_iter, HYBRID=0)

        kmeans = KMeans(n_clusters=Nc_i, random_state=0, max_iter=n_iter, tol=0.0001).fit(data)
        best_fit_kmeans = fitness_Kmeans(kmeans, data, Nc_i, d_L2)

        [best_fit_HYBRID, d] = PSO(data=data, Nc=Nc_i, Nd=Nd,bounds=bounds, n_particles=n_particles, n_iter=n_iter, HYBRID=1)

        besties[0][Nc_i-np.min(Nc)][sim_i] = best_fit_PSO
        besties[1][Nc_i-np.min(Nc)][sim_i] = best_fit_kmeans
        besties[2][Nc_i-np.min(Nc)][sim_i] = best_fit_HYBRID


# Statistics
avg = np.mean(besties, axis=2)

# Plotting
if PROBLEM ==1:
    for i in range(3):
        plt.errorbar(Nc, avg[i])
    plt.legend(["PSO", "Kmeans", "Hybrid"])
    plt.xlabel("# Clusters")
    plt.ylabel("Fitness")
    plt.title("Minimization performance on Artificial data 1")
    plt.tight_layout()
if PROBLEM ==0:
    plt.bar(range(len(avg)), avg, color=[0.2, 0.2, 0.8])
    plt.xticks(range(len(avg)), ("PSO", "Kmeans", "Hybrid"))
    plt.ylabel("Fitness")
    plt.title("Minimization performance on Iris data using 3 clusters")
    plt.tight_layout()

plt.show()
