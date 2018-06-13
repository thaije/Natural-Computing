import gym_super_mario_bros
from matplotlib import pyplot as plt
import numpy as np
import copy
import pickle as pkl
import random
from scipy.misc import imresize


class Pool():

	def __init__(self):
	 
		self.species = []
		self.generation = 0
		self.innovation = 0
		self.currentSpecies = 0
		self.currentGenome = 0
		self.currentFrame = 0
		self.maxFitness = 0

class Species():

	def __init__(self):
			
		self.topFitness = 0
		self.staleness = 0
		self.genomes = []
		self.averageFitness = 0        

class Genome():

	def __init__(self, MutateConnectionsChance=.25, LinkMutationChance=2., BiasMutationChance=.5, NodeMutationChance=.5, EnableMutationChance=.2, DisableMutationChance=.4, StepSize=.1):

		self.genes = []
		self.fitness = 0
		self.adjustedFitness = 0
		self.network = None
		self.maxneuron = 0
		self.globalRank = 0
		self.mutationRates = {}
		self.mutationRates["connections"] = MutateConnectionsChance
		self.mutationRates["link"] = LinkMutationChance
		self.mutationRates["bias"] = BiasMutationChance
		self.mutationRates["node"] = NodeMutationChance
		self.mutationRates["enable"] = EnableMutationChance
		self.mutationRates["disable"] = DisableMutationChance
		self.mutationRates["step"] = StepSize        
	
	def copy(self, orig):

		self.genes = copy.deepcopy(orig.genes)
		self.maxneuron = orig.maxneuron
		self.mutationRates["connections"] = orig.mutationRates["connections"]
		self.mutationRates["link"] = orig.mutationRates["link"]
		self.mutationRates["bias"] = orig.mutationRates["bias"]
		self.mutationRates["node"] = orig.mutationRates["node"]
		self.mutationRates["enable"] = orig.mutationRates["enable"]
		self.mutationRates["disable"] = orig.mutationRates["disable"]


class Gene():

	def __init__(self):

		self.into = 0
		self.out = 0
		self.weight = 0.0
		self.enabled = True
		self.innovation = 0

class Neuron():

	def __init__(self):
		self.incoming = []
		self.value = 0.0

class Network():

	def __init__(self, genome, inp_size=[224, 256, 1], out_size=14, max_nodes=1000000):

		self.inp_size = np.prod(inp_size) + 1#bias
		self.out_size = out_size
		self.max_nodes = max_nodes

		self.neurons = {i: Neuron() for i in range(self.inp_size)}
		for o in range(self.out_size):
			self.neurons[max_nodes+o] = Neuron()

		genome.genes.sort(key=lambda gene: gene.out, reverse=True)
			
		for gene in genome.genes:
			if gene.enabled:
				if not gene.out in self.neurons:
					self.neurons[gene.out] = Neuron()
	 
				neuron = self.neurons[gene.out]
				neuron.incoming.append(gene)
				if not gene.into in self.neurons:
					self.neurons[gene.into] = Neuron()

		genome.network = self        

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def evaluateNetwork(network, inputs):
	inputs = np.append(inputs,[1])
	if len(inputs) != network.inp_size:
		print('Incorrect number of neural network inputs.')
		return 

	for i,inp in enumerate(inputs):
		network.neurons[i].value = inputs[i]

	for	i,neuron in network.neurons.items():
		out = 0
		for i,inc in enumerate(neuron.incoming):
			next = network.neurons[inc.into]
			out += inc.weight*next.value
			print(out)
		if out:
			neuron.value = sigmoid(out)
			print(neuron.value)

	outputs = []
	for o in range(network.out_size):
		outputs.append(network.neurons[network.max_nodes+o].value)
	
	return outputs

def crossover(g1, g2):
	if g2.fitness > g1.fitness:
		g1,g2 = g2,g1

	child = Genome()
	innovations = {}
	for g in g2.genes:
		innovations[g.innovation] = g

	for g in g1.genes:
		other = innovations[g.innovation]
		if other and np.random.rand() > 0.5 and other.enabled:
			child.genes.append(copy.deepcopy(other))
		else:
			child.genes.append(copy.deepcopy(g))

	child.maxneuron = np.max(g1.maxneuron, g2.maxneuron)
	for mutation, rate in g1.mutationRates.items():
		child.mutationRates[mutation] = rate

	return child	 		

def randomNeuron(genes, inp_size, out_size, inp_node, max_nodes=1000000):
	neurons = {}
	if inp_node:
		for i in range(inp_size):
			neurons[i] = True

	for o in range(out_size):
		neurons[max_nodes+o] = True

	for gene in genes:
		if inp_node or gene.into > inp_size:
			neurons[gene.into] = True
		if inp_node or gene.out > inp_size:
			neurons[gene.out] = True

	return random.choice(list(neurons.items()))[0]
	
 
def containsLink(genes, link):
	for g in genes:
		if g.into == link.into and gene.out == link.out:
			return True 

	return False
 
def pointMutate(genome, perturb_chance):
	step = genome.mutationRates['step']
	for gene in genome.genes:
		if math.random.rand() < perturb_chance:
			gene.weight = gene.weight + 2*step*(np.random.rand() - 0.5)
		else:
			gene.weight = 4*(np.random.rand() - 0.5)	
 
def linkMutate(genome, forceBias, inp_size, out_size, pool):

	neuron1 = randomNeuron(genome.genes,inp_size, out_size, True)
	neuron2 = randomNeuron(genome.genes,inp_size, out_size, False)

	newLink = Gene()
	if neuron1 <= inp_size and neuron2 <= inp_size:
		return
	if neuron2 <= inp_size:
		neuron2, neuron1 = neuron1, neuron2

	newLink.into = neuron1
	newLink.out = neuron2
	if forceBias:
		newLink.into = inp_size

	if containsLink(genome.genes, newLink):
		return

	
	pool.innovation += 1
	newLink.innovation =  pool.innovation
	newLink.weight = 4*(np.random.rand() - 0.5)
	genome.genes.append(newLink)
	


def nodeMutate(genome, pool):
	if not len(genome.genes):
		return


	genome.maxneuron += 1
	gene = genome.genes[np.random.randint(len(genome.genes))]	
	
	if not gene.enabled:
		return

	gene.enabled = False

	gene1 = copy.deepcopy(gene)
	gene1.out = genome.maxneuron
	gene1.weight = 1.
	pool.innovation += 1
	gene1.innovation = pool.innovation
	gene1.enabled = True
	genome.genes.append(gene1)

	
	gene2 = copy.deepcopy(gene)
	gene2.into = genome.maxneuron
	pool.innovation += 1
	gene2.innovation = pool.innovation
	gene2.enabled = True
	genome.genes.append(gene2)


def enableDisableMutate(genome, enable):

	candidates = []
	for gene in genome.genes:
		if gene.enabled != enable:
			candidates.append(gene)
	if not len(candidates):
		return

	gene = candidates[np.random.randint(len(candidates))]
	gene.enabled = not gene.enabled


def determine_mutate(genome, p, mutation_func, **kwargs):
	while p > 0:
		if np.random.rand() < p:
			#pdb.set_trace()
			mutation_func(genome, **kwargs)
		p -= 1

def mutate(genome, inp_size, out_size, pool, perturb_chance):
	for mutation,rate in genome.mutationRates.items():
		if np.random.rand() > .5:
			genome.mutationRates[mutation] = .95*rate
		else:
			genome.mutationRates[mutation] = 1.04263*rate

	if np.random.rand() < genome.mutationRates["connections"]:
		pointMutate(genome, perturb_chance)	

	arguments = {'forceBias': False, 'inp_size': inp_size, 'out_size': out_size, 'pool': pool}
	determine_mutate(genome, genome.mutationRates['link'], linkMutate, **arguments)

	arguments = {'forceBias': True, 'inp_size': inp_size, 'out_size': out_size, 'pool': pool}
	determine_mutate(genome, genome.mutationRates['bias'], linkMutate, **arguments)
			
	arguments = {'pool': pool}
	determine_mutate(genome, genome.mutationRates['node'], nodeMutate, **arguments)

	arguments = {'enable': True}
	determine_mutate(genome, genome.mutationRates['enable'], enableDisableMutate, **arguments)
	   
	arguments = {'enable': False}
	determine_mutate(genome, genome.mutationRates['disable'], enableDisableMutate, **arguments)


def disjoint(genes1, genes2):
	i1 = {}
	for gene in genes1:
		i1[gene.innovation] = True

	i2 = {}
	for gene in genes2:
		i2[gene.innovation] = True	

	disjointGenes = 0
	for i,gene in enumerate(genes1):
		if not gene.innovation in i2:
			disjointGenes += 1

	for i,gene in enumerate(genes2):
		if not gene.innovation in i1:
			disjointGenes += 1		
	
	return disjointGenes / np.max((len(genes1),len(genes2)))

def weights(genes1, genes2):
	i2 = {}
	for gene in genes2:
		i2[gene.innovation] = gene

	sum = 0
	coincident = 0
	for i,gene in enumerate(genes1):
		try:
			gene2 = i2[gene.innovation]
			sum += np.abs(gene.weight - genes2.weight)
			coincident += 1
		except KeyError:
			pass

	if not coincident:
		return np.inf

	return sum/coincident

def sameSpecies(genome1, genome2, delta_disjoint, delta_weight, delta_threshold):
	dd = delta_disjoint*disjoint(genome1.genes, genome2.genes)
	dw = delta_weight*weights(genome1.genes, genome2.genes)
	return dd + dw < delta_threshold
	   
def rankGlobally(pool):
	glob = []
	for species in pool.species:
		for genome in species.genomes:
			glob.append(genome)

	glob.sort(key=lambda x: x.fitness, reverse=True)
	for i,g in enumerate(glob):
		g.globalRank = i


def calculateAverageFitness(species):
	total = 0
	for g in species.genomes:
		total += genome.globalRank

	species.averageFitness = total / len(species.genomes)


def totalAverageFitness(pool):
	total = 0
	for s in pool.species:
		total += species.averageFitness

	return total		
 
def cullSpecies(pool, cutToOne):
	for s in pool.species:
		species.genomes.sort(key=lambda g: g.fitness)
		remaining = np.ceil(len(species.genomes)//2)

		if cutToOne:
			remaining = 1

		while len(species.genomes) > remaining:
				species.genomes.remove(species.genomes[-1])

def breedChild(species, crossover_chance):
	
	if np.random.rand() < crossover_chance:
		g1 = species.genomes[np.random.randint(len(species.genomes))]
		g2 = species.genomes[np.random.randint(len(species.genomes))]
		child = crossover(g1,g2)
	else:
		g = species.genomes[np.random.randint(len(species.genomes))]	
		child = Genome().copy(g)

	mutate(child)
	
	return child	


def removeStaleSpecies(pool, stale_species):
	survived = []
	for species in pool.species:
		species.genomes.sort(key=lambda x: x.fitness)

		if species.genomes[0].fitness > species.topFitness:
			species.topFitness = genomes[0].fitness
			species.staleness = 0
		else:
			species.staleness += 1
		if species.staleness < stale_species or species.topFitness >= pool.maxFitness:
			survived.append(species)			

	pool.species = survived


def removeWeakSpecies(pool, population):
	survived = []
	sum = totalAverageFitness()
	for species in pool.species:
		breed = species.averageFitness // (sum*population)
		if breed >=1:
			survived.append(species)

	pool.species = survived		
 
def addToSpecies(pool, child, delta_disjoint, delta_weight, delta_threshold):
	found = False
	for species in pool.species:
		if not found and sameSpecies(child, species.genomes[0], delta_disjoint, delta_weight, delta_threshold):
			species.genomes.append(child)
			found = True

	if not found:
		childSpecies = Species()
		childSpecies.genomes.append(child)
		pool.species.append(childSpecies)


def newGeneration(pool, stale_species, population, crossover_chance):
	cullSpecies(pool, False)
	rankGlobally(pool)
	removeStaleSpecies(pool, stale_species)
	rankGlobally(pool)

	for species in pool.species:
		calculateAverageFitness(species)

	removeWeakSpecies(pool, population)
	sum = totalAverageFitness()
	children = []
	for species in pool.species:
		breed = species.averageFitness // (sum * population) - 1
		for i in range(breed):
			children.append(breedChild(species, crossover_chance))

	cullSpecies(pool,True)			

	while len(children) + len(pool.species) < population:
		species = pool.species[np.random.randint(len(pool.species))]
		children.append(breedChild(species, crossover_chance))

	for child in children:
		addToSpecies(pool, child)

	pool.generation += 1
			

def initializePool(population, inp_size, out_size, delta_disjoint, delta_weight, delta_threshold, perturb_chance):
	pool = Pool()
	size = np.prod(inp_size) + 1
	for i in range(population):
		basic = Genome()
		basic.maxneuron = size
		mutate(basic, size, out_size, pool, perturb_chance)
		addToSpecies(pool, basic, delta_disjoint, delta_weight, delta_threshold)

	initializeRun(pool, inp_size)

	return pool


def initializeRun(pool, inp_size):
	pool.currentFrame = 0
	species = pool.species[pool.currentSpecies]
	genome = species.genomes[pool.currentGenome]
	net = Network(genome, inp_size)
 

def evaluateCurrent(pool, inputs):
	species = pool.species[pool.currentSpecies]
	genome = species.genomes[pool.currentGenome]

	controller = evaluateNetwork(genome.network, inputs)

	return controller


def nextGenome(pool,stale_species,population,crossover_chance): 
	pool.currentGenome += 1
	if pool.currentGenome >= len(pool.species[pool.currentSpecies].genomes):
		pool.currentGenome = 0
		pool.currentSpecies += 1
		if pool.currentSpecies >= len(pool.species):
			newGeneration(pool, stale_species, population, crossover_chance)
			pool.currentSpecies = 0


def fitnessAlreadyMeasured(pool):
	species = pool.species[pool.currentSpecies]
	genome = species.genomes[pool.currentGenome]
	return genome.fitness != 0
 
def savePool(pool,filename='pool.pkl'):
		pkl.dump(pool, open(filename, 'wb'))

def loadPool(filename='pool.pkl'):    
	return pkl.load(open(filename,'rb'))    

if __name__ == '__main__':

	env = gym_super_mario_bros.make('SuperMarioBros-v2')

	inp_size = [150, 180, 1]
	out_size = 14 
	population = 300
	delta_disjoint = 2.0
	delta_weight = 0.4
	delta_threshold = 1.0
	 
	stale_species = 15

	MutateConnectionsChance = 0.25
	perturb_chance = 0.90
	crossover_chance = 0.75
	LinkMutationChance = 2.0
	NodeMutationChance = 0.50
	BiasMutationChance = 0.40
	StepSize = 0.1
	DisableMutationChance = 0.4
	EnableMutationChance = 0.2

	pool = initializePool(population, inp_size, out_size, delta_disjoint, delta_weight, delta_threshold, perturb_chance)
	#pool = loadPool('pool_generation0.pkl')

	def rgb2gray(rgb):
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

	def update_pos(reward, info):
		info['pos'].append(reward)
		info['curr'] = np.sum(info['pos'])
		if info['curr']  > info['best'] :
			info['best']  = info['curr'] 

	pos_info = {'pos':[], 'curr': 0, 'best': 0}



	LEVEL_LEN = 3186 #not sure if correct

	while True:
		time_out = 20
		state = env.reset()
		initializeRun(pool, inp_size)
		state, reward, done, info = env.step(env.action_space.sample())
		print(state.shape)
		while not done:
			print(time_out)
			species = pool.species[pool.currentSpecies]
			genome = species.genomes[pool.currentGenome]

			if not (pool.currentFrame%5):
				controller = evaluateCurrent(pool,imresize(rgb2gray(state),inp_size).flatten())

			probs = controller/np.sum(controller)
			if np.any(np.isnan(probs)):
				probs = None	
			action = np.random.choice(range(len(controller)),p=probs)
			state, reward, done, info = env.step(action)
			update_pos(reward, pos_info)
			if pos_info['curr'] < pos_info['best']:
				time_out -= 1
			if time_out + pool.currentFrame//4 < 0:
				done = True	

			if done:
				state = env.reset()
				
				fitness = pos_info['best'] - pool.currentFrame/2
				
				if pos_info['best'] > LEVEL_LEN:
					fitness += 1000

				if not fitness:
					fitness = -1

				genome.fitness = fitness

				if fitness > pool.maxFitness:
					pool.maxFitness = fitness
					savePool(pool, 'pool_generation{}.pkl'.format(pool.generation))	

				print(("Gen " + str(pool.generation) + " species " + str(pool.currentSpecies) + " genome " + str(pool.currentGenome) + " fitness: " + str(fitness)))

				pool.currentSpecies = 0
				pool.currentGenome = 0
				while fitnessAlreadyMeasured(pool):
					nextGenome(pool, stale_species, population, crossover_chance)
				
				pos_info = {'pos':[], 'curr': 0, 'best': 0}

		 

					   
			pool.currentFrame = pool.currentFrame + 1

		 

	env.close()