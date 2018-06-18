import gym_super_mario_bros
from matplotlib import pyplot as plt
import numpy as np
import copy
import pickle as pkl
import random
import pdb
import sys
from scipy.misc import imresize


class Pool():
	'''
	Class holding all evolved species, responsible for evolution
	'''
	def __init__(self, inp_size, out_size):
		'''
		Constructor
		:param inp_size (int) number of input nodes
		:param out_size (int) number of output nodes
		'''
		self.species = []
		self.generation = 0
		self.innovation = 0
		self.current_species = 0
		self.current_genome = 0
		self.current_frame = 0
		self.max_fitness = 0
		self.inp_size = inp_size
		self.out_size = out_size

	def total_average_fitness(self):
		'''
		Calculates total fitness of all species in pool
		:return total sum of fitnesses
		'''
		total = 0
		for s in self.species:
			total += s.average_fitness
		return total		

	def reduce_species(self, highlander):
		'''
		Removes species' genomes with lowest fitness until only half the population remains
		:param highlander (bool) if true only one species remains
		'''
		for s in self.species:
			species.genomes.sort(key=lambda g: g.fitness)
			remaining = np.ceil(len(species.genomes)/2)

			if highlander:
				remaining = 1

			while len(species.genomes) > remaining:
					species.genomes.remove(species.genomes[-1])	

	def remove_weak_species(self, population):
		'''
		Removes species with low fitness such that approx population species survive
		:param population (int) number of species to remain
		'''
		survived = []
		total = self.total_average_fitness()
		for species in self.species:
			breed = np.floor(species.average_fitness / total * population)
			if breed >=1:
				survived.append(species)

		if not len(survived):
			survived = [self.species[self.current_species]]		
			
		self.species = survived		
	 
	def add_to_species(self, child, delta_disjoint, delta_weight, delta_threshold):
		'''
		Adds child species to pool if same architecture not already in pool
		:param child (Species) to add
		:param delta_disjoint (float) distance weight of disjoint genes for counting species as equal
		:param delta_weight (float) weight for weight differences
		:param delta_treshold (float) distance threshold at which species count as distinct
		'''
		found = False
		for species in self.species:
			if not found and same_species(child, species.genomes[0], delta_disjoint, delta_weight, delta_threshold):
				species.genomes.append(child)
				found = True

		if not found:
			child_species = Species()
			child_species.genomes.append(child)
			self.species.append(child_species)		

	def remove_stale_species(self, stale_species):
		'''
		Removes species that have not fitness has not increased for too long
		:param stale_species (int) iteration count after which to remove species if no improvement
		'''
		survived = []
		for species in self.species:
			if not len(species.genomes):
				continue

			species.genomes.sort(key=lambda x: x.fitness)

			if species.genomes[0].fitness > species.top_fitness:
				species.top_fitness = species.genomes[0].fitness
				species.staleness = 0
			else:
				species.staleness += 1
			if species.staleness < stale_species or species.top_fitness >= self.max_fitness:
				survived.append(species)			

		self.species = survived		

	def rank_globally(self):
		'''
		Calculates ranking according to fitness of all genomes of all species in pool
		'''
		glob = []
		for species in self.species:
			for genome in species.genomes:
				glob.append(genome)

		glob.sort(key=lambda x: x.fitness, reverse=True)
		for i,g in enumerate(glob):
			g.global_rank = i		

	def new_generation(self, stale_species, population, crossover_chance, perturb_chance, delta_disjoint, delta_weight, delta_threshold):
		'''
		Produce new generation of species by removing weak species and creating new through crossover and mutation
		:param stale_species (int) when to remove species because of no improvement
		:param population (int) desired population size after removing weak species
		:param crossover_chance (float) probability of crossover
		:param perturb_chance (float) probability to change random connection weight
		:param delta_disjoint (float) distance weight of disjoint genes for counting species as equal
		:param delta_weight (float) weight for weight differences
		:param delta_treshold (float) distance threshold at which species count as distinct
		'''
		self.reduce_species(False)
		self.rank_globally()
		self.remove_stale_species(stale_species)
		self.rank_globally()

		for species in self.species:
			species.calculate_average_fitness()

		self.remove_weak_species(population)
		sum = self.total_average_fitness()
		children = []
		for species in self.species:
			breed = np.floor(species.average_fitness / sum * population) - 1
			for i in range(int(breed)):
				children.append(breed_child(self, species, crossover_chance, perturb_chance))

		self.reduce_species(True)			

		while len(children) + len(self.species) < population:
			species = self.species[np.random.randint(len(self.species))]
			children.append(breed_child(self, species, crossover_chance, perturb_chance))

		for child in children:
			self.add_to_species(child, delta_disjoint, delta_weight, delta_threshold)

		self.generation += 1	

	def next_genome(self, stale_species, population, crossover_chance, perturb_chance, delta_disjoint, delta_weight, delta_threshold): 
		'''
		Choose next genome for evaluation and create new generation if all species have been evaluated
		:param stale_species (int) when to remove species because of no improvement
		:param population (int) desired population size after removing weak species
		:param crossover_chance (float) probability of crossover
		:param perturb_chance (float) probability to change random connection weight
		:param delta_disjoint (float) distance weight of disjoint genes for counting species as equal
		:param delta_weight (float) weight for weight differences
		:param delta_treshold (float) distance threshold at which species count as distinct
		'''
		self.current_genome += 1
		if self.current_genome >= len(self.species[self.current_species].genomes):
			self.current_genome = 0
			self.current_species += 1
			if self.current_species >= len(self.species):
				self.new_generation(stale_species, population, crossover_chance, perturb_chance, delta_disjoint, delta_weight, delta_threshold)
				self.current_species = 0			

class Species():
	'''
	Class representing a species of evolved networks
	'''
	def __init__(self):
		'''
		Constructor
		'''
		self.top_fitness = 0
		self.staleness = 0
		self.genomes = []
		self.average_fitness = 0    

	def calculate_average_fitness(self):
		'''
		Calculates average fitness of all genomes within species
		'''
		total = 0
		for g in self.genomes:
			total += g.global_rank

		self.average_fitness = total / len(self.genomes)	    

class Genome():
	'''
	Class representing a genome that encodes a neural network architecture
	'''
	def __init__(self, mutate_connections_chance=.25, link_mutation_chance=2., bias_mutation_chance=.5, node_mutation_chance=.5, enable_mutation_chance=.2, disable_mutation_chance=.4, step_size=.1):
		'''
		Constructor
		:param mutate_connections_chance (float) probability perturbe connection between neurons
		:paran link_mutation_chance (float) probability to form new connection between neurons
		:param bias_mutation_chance (float) probability to create new bias connection weight
		:param node_mutation_chance (float) probability to create new artificial neuron
		:param enable_mutation_chance (float) probability to enable disabled neuron
		:param disable_mutation_chance (float) probability to disable enabled neuron
		:param step_size (float) magnitude of random weight perturbation
		'''
		self.genes = []
		self.fitness = 0
		self.adjusted_fitness = 0
		self.network = None
		self.maxneuron = 0
		self.global_rank = 0
		self.mutation_rates = {}
		self.mutation_rates["connections"] = mutate_connections_chance
		self.mutation_rates["link"] = link_mutation_chance
		self.mutation_rates["bias"] = bias_mutation_chance
		self.mutation_rates["node"] = node_mutation_chance
		self.mutation_rates["enable"] = enable_mutation_chance
		self.mutation_rates["disable"] = disable_mutation_chance
		self.mutation_rates["step"] = step_size        
	
	def copy(self, orig):
		'''
		Copy 'Constructor' - takes genome and makes self a copy thereof
		:param orig (Genome) genome to make copy of
		'''
		self.genes = copy.deepcopy(orig.genes)
		self.maxneuron = orig.maxneuron
		self.mutation_rates["connections"] = orig.mutation_rates["connections"]
		self.mutation_rates["link"] = orig.mutation_rates["link"]
		self.mutation_rates["bias"] = orig.mutation_rates["bias"]
		self.mutation_rates["node"] = orig.mutation_rates["node"]
		self.mutation_rates["enable"] = orig.mutation_rates["enable"]
		self.mutation_rates["disable"] = orig.mutation_rates["disable"]


	def contains_link(self, link):
		'''
		Computes whether weight between nodes is already encoded in genome
		:param link (gene) describes connection between neurons
		:return whether weight already in genome
		'''
		for g in self.genes:
			if g.into == link.into and g.out == link.out:
				return True 

		return False

	def get_random_neuron(self, inp_size, out_size, inp_node, max_nodes=1000000):
		'''
		Returns random neuron from genome
		:param inp_size (int) size of input layer
		:param out_size (int) size of output layer
		:param inp_node (bool) whether to return input nodes
		:param max_nodes (int) maximum number of hidden nodes - denotes output nodes
		'''
		neurons = {}
		if inp_node:
			for i in range(inp_size):
				neurons[i] = True

		for o in range(out_size):
			neurons[max_nodes+o] = True

		for gene in self.genes:
			if inp_node or gene.into > inp_size:
				neurons[gene.into] = True
			if inp_node or gene.out > inp_size:
				neurons[gene.out] = True

		return random.choice(list(neurons.items()))[0]

	def point_mutate(self, perturb_chance):
		'''
		Randomly perturbs connection weight
		:param perturb_chance (float) chance to perturb individual weights
		'''
		step = self.mutation_rates['step']
		for gene in self.genes:
			if np.random.rand() < perturb_chance:
				gene.weight = gene.weight + 2*step*(np.random.rand() - 0.5)
			else:
				gene.weight = 4*(np.random.rand() - 0.5)	
 
	def link_mutate(self, force_bias, pool):
		'''
		Randomly creates new connection between neurons
		:param force_bias (bool) if true new bias weight will be created
		:param pool (Pool) needed for innovation number
		'''
		neuron1 = self.get_random_neuron(pool.inp_size, pool.out_size, True)
		neuron2 = self.get_random_neuron(pool.inp_size, pool.out_size, False)

		new_link = Gene()
		if neuron1 <= pool.inp_size and neuron2 <= pool.inp_size:
			return
		if neuron2 <= pool.inp_size:
			neuron2, neuron1 = neuron1, neuron2

		new_link.into = neuron1
		new_link.out = neuron2
		if force_bias:
			new_link.into = pool.inp_size

		if self.contains_link(new_link):
			return

		pool.innovation += 1
		new_link.innovation = pool.innovation
		new_link.weight = 4*(np.random.rand() - 0.5)
		self.genes.append(new_link)
	
	def node_mutate(self, pool):
		'''
		Randomly creates new network node
		:param pool (Pool) needed for innovation number
		'''
		if not len(self.genes):
			return

		self.maxneuron += 1
		gene = self.genes[np.random.randint(len(self.genes))]	
		
		if not gene.enabled:
			return

		gene.enabled = False

		gene1 = copy.deepcopy(gene)
		gene1.out = self.maxneuron
		gene1.weight = 1.
		pool.innovation += 1
		gene1.innovation = pool.innovation
		gene1.enabled = True
		self.genes.append(gene1)

		
		gene2 = copy.deepcopy(gene)
		gene2.into = self.maxneuron
		pool.innovation += 1
		gene2.innovation = pool.innovation
		gene2.enabled = True
		self.genes.append(gene2)


	def enable_disable_mutate(self, enable):
		'''
		Randomly disables or enables gene
		:param enable (bool) if true enable gene else disable
		'''
		candidates = []
		for gene in self.genes:
			if gene.enabled != enable:
				candidates.append(gene)
		if not len(candidates):
			return

		gene = candidates[np.random.randint(len(candidates))]
		gene.enabled = not gene.enabled


	def determine_mutate(self, p, mutation_func, **kwargs):
		'''
		Helper function to call respective mutation function
		:param p (float) parameter determining how many times mutation is executed
		:param mutation_rates (function) mutation function to be called
		:param kwargs (dict) dictionary of named function arguments
		'''
		while p > 0:
			if np.random.rand() < p:
				mutation_func(**kwargs)
			p -= 1

	def mutate(self, pool, perturb_chance):
		'''
		Executes genome mutations
		:param pool (Pool) needed for innovation number
		:param perturb_chance (float) probability of individual weight perturbation
		'''
		for mutation,rate in self.mutation_rates.items():
			if np.random.rand() > .5:
				self.mutation_rates[mutation] = .95*rate
			else:
				self.mutation_rates[mutation] = 1.04263*rate

		if np.random.rand() < self.mutation_rates["connections"]:
			self.point_mutate(perturb_chance)	

		arguments = {'force_bias': False, 'pool': pool}
		self.determine_mutate(self.mutation_rates['link'], self.link_mutate, **arguments)

		arguments = {'force_bias': True, 'pool': pool}
		self.determine_mutate(self.mutation_rates['bias'], self.link_mutate, **arguments)
				
		arguments = {'pool': pool}
		self.determine_mutate(self.mutation_rates['node'], self.node_mutate, **arguments)

		arguments = {'enable': True}
		self.determine_mutate(self.mutation_rates['enable'], self.enable_disable_mutate, **arguments)
		   
		arguments = {'enable': False}
		self.determine_mutate(self.mutation_rates['disable'], self.enable_disable_mutate, **arguments)


class Gene():
	'''
	Class representing individual gene which in turn encodes weight connections
	'''
	def __init__(self):
		'''
		Constructor
		'''
		self.into = 0
		self.out = 0
		self.weight = 0.0
		self.enabled = True
		self.innovation = 0

class Neuron():
	'''
	Class representing artficial neuron
	'''
	def __init__(self):
		'''
		Constructor
		'''
		self.incoming = []
		self.value = 0.0

class Network():
	'''
	Class representing entire network encoded by genome
	'''
	def __init__(self, genome, inp_size=[224, 256, 1], out_size=14, max_nodes=1000000):
		'''
		Constructor
		:param genome (Genome) represents network
		:param inp_size (array_like) input dimensions
		:param out_size (int) output dimensions
		:param max_nodes (int) maximum number of input and hidden nodes
		'''
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


	def evaluate(self, inputs):
		'''
		Calculates forward pass through the network
		:param inputs (array_like) neural network input
		:returns network output
		'''
		inputs = np.append(inputs,[1])
		if len(inputs) != self.inp_size:
			print('Wrong input dimensions')
			return 

		for i,inp in enumerate(inputs):
			self.neurons[i].value = inputs[i]

		for	i,neuron in self.neurons.items():
			out = 0
			for i,inc in enumerate(neuron.incoming):
				next = self.neurons[inc.into]
				out += inc.weight*next.value
			if out:
				neuron.value = sigmoid(out)

		outputs = []
		for o in range(self.out_size):
			outputs.append(self.neurons[self.max_nodes+o].value)
		
		return outputs	   

def sigmoid(x):
	'''
	Calculates sigmoid function
	'''	
	return 1 / (1 + np.exp(-x))

def crossover(genes1, genes2):
	'''
	Generates crossover between two genomes
	:param genes1 (Genome)
	:param genes2 (Genome)
	:return crossover child
	'''
	if genes2.fitness > genes1.fitness:
		genes1,genes2 = genes2,genes1

	child = Genome()
	innovations = {}
	for g in genes2.genes:
		innovations[g.innovation] = g

	for g in genes1.genes:
		try:
			other = innovations[g.innovation]
		except KeyError:
			other = None	
		if other and np.random.rand() > 0.5 and other.enabled:
			child.genes.append(copy.deepcopy(other))
		else:
			child.genes.append(copy.deepcopy(g))

	child.maxneuron = np.max((genes1.maxneuron, genes2.maxneuron))
	for mutation, rate in genes1.mutation_rates.items():
		child.mutation_rates[mutation] = rate

	return child	 		
	
  
def disjoint(genes1, genes2):
	'''
	Calculates fraction of disjoint genes between two genomes
	:param genes1 (Genome)
	:param genes2 (Genome)
	:return fraction of disjoint genes
	'''
	i1 = {}
	for gene in genes1:
		i1[gene.innovation] = True

	i2 = {}
	for gene in genes2:
		i2[gene.innovation] = True	

	disjoint_genes = 0
	for i,gene in enumerate(genes1):
		if not gene.innovation in i2:
			disjoint_genes += 1

	for i,gene in enumerate(genes2):
		if not gene.innovation in i1:
			disjoint_genes += 1		
	
	return disjoint_genes / np.max((len(genes1),len(genes2)))

def weights(genes1, genes2):
	'''
	Calculates distance between weights of two genomes
	:param genes1 (Genome)
	:param genes2 (Genome)
	:return distance
	'''
	i2 = {}
	for gene in genes2:
		i2[gene.innovation] = gene

	sum = 0
	coincident = 0
	for i,gene1 in enumerate(genes1):
		try:
			gene2 = i2[gene.innovation]
			sum += np.abs(gene1.weight - gene2.weight)
			coincident += 1
		except KeyError:
			pass

	if not coincident:
		return np.inf

	return sum/coincident

def same_species(genome1, genome2, delta_disjoint, delta_weight, delta_threshold):
	'''
	Determines whether two genomes belong to the same species
	:param genes1 (Genome)
	:param genes2 (Genome)
	:param delta_disjoint (float) weight for relevance of disjoint genes
	:param delta_weight (float) weight for relevance between weight distance
	:param delta_threshold (float) threshold for determining whether genomes belong to distinct spiecies
	:return true if same species
	'''
	dd = delta_disjoint*disjoint(genome1.genes, genome2.genes)
	dw = delta_weight*weights(genome1.genes, genome2.genes)
	return dd + dw < delta_threshold
	   
 
def breed_child(pool, species, crossover_chance, perturb_chance):
	'''
	Creates child genome through crossover and mutation between two random genomes
	:param pool (Pool)
	:param species (Species) 
	:param crossover_chance (float) probability of crossover
	:param perturb_chance (float) probability for perturbation for mutate function
	:return child genome
	'''
	if np.random.rand() < crossover_chance:
		g1 = species.genomes[np.random.randint(len(species.genomes))]
		g2 = species.genomes[np.random.randint(len(species.genomes))]
		child = crossover(g1,g2)
	else:
		g = species.genomes[np.random.randint(len(species.genomes))]	
		child = Genome()
		child.copy(g)

	child.mutate(pool, perturb_chance)
	
	return child	

			

def initialize_pool(population, inp_size, out_size, delta_disjoint, delta_weight, delta_threshold, perturb_chance):
	'''
	Initialises new Pool
	:param population (int) number of species per generation
	:param inp_size (array_like) input dimensions
	:param out_size (int) number of output neurons
	:param delta_disjoint (float) weight for relevance of disjoint genes
	:param delta_weight (float) weight for relevance between weight distance
	:param delta_threshold (float) threshold for determining whether genomes belong to distinct spiecies
	:param perturb_chance (float) probability for perturbation for mutate function
	return new Pool
	'''
	size = np.prod(inp_size) + 1
	pool = Pool(size, out_size)
	for i in range(population):
		basic = Genome()
		basic.maxneuron = size
		basic.mutate(pool, perturb_chance)
		pool.add_to_species(basic, delta_disjoint, delta_weight, delta_threshold)

	initialize_run(pool, inp_size)

	return pool


def initialize_run(pool, inp_size):
	'''
	Creates new Network for current evaluation run
	:param pool (Pool)
	:param inp_size (array_like) input dimensions
	'''
	pool.current_frame = 0
	species = pool.species[pool.current_species]
	genome = species.genomes[pool.current_genome]
	net = Network(genome, inp_size)
 

def evaluate_current(pool, inputs):
	'''
	Evaluates current network
	:param pool (Pool)
	:param inputs (array_like) neural network inputs
	:return neural network output
	'''
	species = pool.species[pool.current_species]
	genome = species.genomes[pool.current_genome]

	output = genome.network.evaluate(inputs)

	return output


def genome_evaluated(pool):
	'''
	Returns whether genome has already been evaluated
	'''
	species = pool.species[pool.current_species]
	genome = species.genomes[pool.current_genome]
	return genome.fitness != 0
 
def save_pool(pool,filename='pool.pkl'):
	'''
	Saves pool
	:param pool (Pool)
	:param filename (str)
	'''
	pkl.dump(pool, open(filename, 'wb'))

def load_pool(filename='pool.pkl'):    
	'''
	Loads pool
	:param filename (str)
	:returns loaded pool
	'''
	return pkl.load(open(filename,'rb'))    

def rgb2gray(rgb):
	'''
	Creates grayscale from rgb image
	:param rgb (array_like) image to be transformed
	:return grayscale image
	'''
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def update_pos(reward, info):
	'''
	Updates postion of mario agent
	:param reward (float)
	:param info (dict) previous information
	'''
	info['pos'].append(reward)
	info['curr'] = np.sum(info['pos'])
	if info['curr']  > info['best'] :
		info['best']  = info['curr'] 	

if __name__ == '__main__':

	load = False
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		load = True

	env = gym_super_mario_bros.make('SuperMarioBros-v2')


	perturb_chance = 0.90
	crossover_chance = 0.75
	inp_size = [224, 256, 1]#[150, 180, 1]
	out_size = 14 
	population = 100
	delta_disjoint = 2.0
	delta_weight = 0.4
	delta_threshold = 1.0
	 
	stale_species = 15

	if load:
		pool = load_pool(filename)
	else:
		pool = initialize_pool(population, inp_size, out_size, delta_disjoint, delta_weight, delta_threshold, perturb_chance)
	

	pos_info = {'pos':[], 'curr': 0, 'best': 0}


	LEVEL_LEN = 3186 #not sure if correct

	counter = 0
	while True:
		time_out = 20
		state = env.reset()
		initialize_run(pool, inp_size)
		state, reward, done, info = env.step(env.action_space.sample())
		while not done:
			#print(time_out)
			species = pool.species[pool.current_species]
			genome = species.genomes[pool.current_genome]

			if not (pool.current_frame%5):
				state = state/255
				output = evaluate_current(pool,imresize(rgb2gray(state),inp_size).flatten())

			probs = output/np.sum(output)
			if np.any(np.isnan(probs)):
				probs = None	
			action = np.random.choice(range(len(output)),p=probs)
			state, reward, done, info = env.step(action)
			update_pos(reward, pos_info)

			if pos_info['curr'] < pos_info['best']:
				time_out -= 1

			if pos_info['curr'] <= -100:
				done = True

			#if pos_info['best'] <= pos_info['curr']:
			#	time_out += 1

			if time_out + pool.current_frame//4 < 0:
				done = True	

			if done:
				state = env.reset()
				
				fitness = pos_info['best'] - pool.current_frame/2
				
				if pos_info['best'] > LEVEL_LEN:
					fitness += 1000

				if not fitness:
					fitness = -1

				genome.fitness = fitness

				if fitness > pool.max_fitness:
					pool.max_fitness = fitness
					save_pool(pool, 'pool_generation{}.pkl'.format(pool.generation))	

				print(("Gen " + str(pool.generation) + " species " + str(pool.current_species) + " genome " + str(pool.current_genome) + " fitness: " + str(fitness)+ " maxfitness: " + str(pool.max_fitness)))

				pool.current_species = 0
				pool.current_genome = 0
				while genome_evaluated(pool):
					pool.next_genome(stale_species, population, crossover_chance, perturb_chance, delta_disjoint, delta_weight, delta_threshold)
				
				pos_info = {'pos':[], 'curr': 0, 'best': 0}

					   
			pool.current_frame = pool.current_frame + 1

		#counter += 1
		#if not counter%100:
		#	save_pool(pool, 'pool_intermediate.pkl')
		 

	env.close()