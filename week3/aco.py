import numpy as np 
from matplotlib import pyplot as plt
from multiprocessing.dummy import Pool
import pickle
import time


class Sudoku:
	'''
	Class representing a field of Sudoku 
	'''

	def __init__(self,field):
		'''
		Constructor
		:param field
			(array_like) initial game
		'''
		self.field = np.copy(field)
		self.length = len(self.field)
		#saves number of digits that can be still entered into a cell
		self.num_possible = np.full(self.field.shape,-1)
		self.calc_possibles()

	def calc_possibles(self):
		'''
		Calculated the number of digits that
		can still legally be entered into a cell
		'''
		for x,y in zip(*np.where(self.field==0)):
			self.num_possible[x,y] = len(self.get_candidates(x,y))

	def get_subfields(self,field,x,y):
		'''
		Returns row, column, and subgrid corresponding to a given cell
		:param x
			(int) x-coordinate of the cell
		:param y
			(int) y-coordinate of the cell
		:returns
			(list) contains row, column, and subgrid	
		'''
		n = int(np.sqrt(self.length))
		row = field[x,:]
		column = field[:,y]

		x_sub = x // n * n
		y_sub = y // n * n
		grid = field[x_sub:x_sub+n,y_sub:y_sub+n].flatten()

		return [row,column,grid]

	def get_candidates(self, x, y):
		'''
		Returns digits that can still be entered into a cell
		:param x
			(int) x-coordinate of the cell
		:param y
			(int) y-coordinate of the cell
		:returns
			(array_like) digits that can still be entered into a cell
		'''
		return np.setdiff1d(np.arange(1,self.length+1), np.unique(self.get_subfields(self.field,x,y)))

	def count_fraction(self, z):
		'''
		Returns count of how often a certain value can still be entered into the entire field
		:param z
			(int) digit value
		:returns
			(int) total allowed number of digit (n) - current count	
		'''
		return (self.length-np.count_nonzero(self.field==z))

	def possible_counts(self):
		'''
		Returns per cell that was not initally field how many digits can still be entered
		'''
		return self.num_possible[self.num_possible>-1]

	def set_value(self,x,y,value):
		'''
		Writes a digit into a cell and recalculates counts for repective row, columnm and subgrid
		:param x
			(int) x-coordinate of the cell
		:param y
			(int) y-coordinate of the cell
		:param value
			(int) digit to be written into cell 
		'''
		if value:
			self.field[x,y] = value
			#if value is set, then the field is definitely blocked
			self.num_possible[x,y] = -1
			fields = self.get_subfields(self.num_possible,x,y) 	
			#subtract possibilities from all "adjacent" cells
			#thereby the possibility count can drop to zero, i.e. the sudoku is no longer solvable	
			for sub in fields:
				sub[sub>0] -= 1
		else:
			self.num_possible[x,y] = -1

class Ant:
	'''
	Class representing an individual ant for ant colony optimization
	'''

	##Class variables
	#side length
	n = 9 
	#pheromone store for all ants
	pheromones = np.random.rand(n,n,n)/10
	#best solution found so far
	best_len = 0
	best = None

	def __init__(self, sudoku):
		'''
		Constructor
		
		:param sudoku
			(array_like) initial sudoku field
		'''
		self.sudoku = sudoku
		self.reset()
		

	def reset(self):
		'''
		Function to reset the ant to its initial state

		:returns self
		'''

		#create new Sudoko
		self.current = Sudoku(self.sudoku)
		#store indeces of unfilled cells
		self.idx = np.array(np.where(self.sudoku==0))
		#store number of cells filled
		self.path_length = np.count_nonzero(self.sudoku > 0)

		return self

	def choose_cell(self):
		'''
		Choose a random cell to fill next

		:returns x,y - coordinates of cell
		'''

		#probability inversely proportional to the number of digits that can still be entered
		#into the cell, i.e. cells with only a few remaining options are chosen first
		p = 1/(self.current.possible_counts()+1e-2)
		which = np.random.choice(np.arange(len(self.idx[0])),p=p/np.sum(p))
		x,y = self.idx[0][which], self.idx[1][which]
		#same cell cannot be picked again at later moment
		self.idx = np.delete(self.idx,which,1)
		return x,y

	def populate(self):
		'''
		Choose and fill a cell with a new digit
		'''

		#choose cell
		x,y = self.choose_cell()
		#get digits that can still be legally entered into the cell
		candidates = self.current.get_candidates(x,y)
		if len(candidates):
			#choose a digit proportional to pheromone count and occurences of the digit in the field
			#digits that have not been used often are chosen with higher probability
			p = [z*self.current.count_fraction(z) for i,z in enumerate(Ant.pheromones[x,y]) if (i+1) in candidates]
			value = np.random.choice(candidates, p=p/np.sum(p))
			self.current.set_value(x,y,value)
			#increase number of filled digits
			self.path_length += 1
		else:
			self.current.set_value(x,y,0)
	

	def is_ready(self):
		'''
		Returns whether ant still can do things
		'''
		return not self.idx[0].size

	def is_solved(self):
		'''
		Returns whether ant has solved the sudoku
		'''
		return self.path_length == len(self.sudoku)**2

	def run(self):
		'''
		Start the ant

		:returns whether the sudoku is solved or not
		'''
		while not self.is_ready():
			self.populate()
		return self.is_solved()

	@classmethod
	def update_pheromones(cls, rho, q, ants):
		'''
		Class method to update pheromones for all ants

		:param rho
			(float) decay parameter
		:param q
			(float) learning constant 
		:param ants
			(array_like) list of all ants used in aco	
		'''
		counts = np.zeros(cls.pheromones.shape)
		
		#find best ant
		best = np.argmax([a.path_length for a in ants])
		best_len = ants[best].path_length
		#save best ant
		if best_len > cls.best_len:
			cls.best = ants[best].current.field
			cls.best_len = best_len

		#accumulate pheromone of all ants on the respective cells
		#by going over all digits
		for z in range(cls.n):
			#and all ants
			for i,ant in enumerate(ants):
				#and adding pheromone to the fields were the ant had this number
				#the pheromone is added proportional to the length of the found solution
				counts[...,z] += q*(ant.path_length/cls.n**2)*(ant.current.field==z+1)
			#the pheromone of the best performing ant is added again	
			counts[...,z] += q*(cls.best_len/cls.n**2)*(cls.best==z+1)

		#add pheromone and decay previous amount
		for (x,y,z), tau in np.ndenumerate(cls.pheromones):
			cls.pheromones[x,y,z] = min(10,max(0.1,(1-rho)*tau + rho*counts[x,y,z]))
		
	@classmethod
	def reset_class(cls):
		'''
		Class method to reset class variables
		'''
		cls.pheromones = np.random.rand(cls.n,cls.n,cls.n)/10
		cls.best_len = 0
		cls.best = None


def read_grid(filename):
	'''
	Method to read a sudoku from txt file
	'''
	with open(filename,'r') as f:
		sudoku = np.loadtxt(f)
	return sudoku



def run_aco(sudoku, n_ants, q, rho, iter):
	'''
	Method to run ant colony optimization algorithm

	:param sudoku
		(array_like) initial sudoku
	:param n_ants
		(int) number of ants 
	:param q
		(float) learning constant
	:param rho
		(float) decay rate
	:param iter
		(float)	number of iterations/cycles 

	:returns best solution, length of solution, list of all lengths, list of best fitnesses		
	'''
	ants = [Ant(sudoku) for i in range(n_ants)]

	path_lengths = []
	path_bests =  []
	for i in range(iter):
		res = [a.run() for a in ants]
		Ant.update_pheromones(rho,q,ants)
		paths = [a.path_length for a in ants]
		path_lengths.extend(paths)
		path_bests.append(np.max(paths)/len(sudoku)**2)
		if any(res):
			break
		else:		
			ants = [a.reset() for a in ants]

	res = (Ant.best, Ant.best_len, path_lengths, path_bests)
	Ant.reset_class()
	return res

if __name__=='__main__':

	np.random.seed(42)

	#read fields
	fields = ['s10a.txt','s10a.txt','s11a.txt','s11a.txt']
	sudoku = read_grid(fields[0])
	n_ants = 5
	
	##Try algorithm with default parameters
	t = time.time()
	solution, l, bests, lengths = run_aco(sudoku, n_ants, 1, .5, 30)
	print('Time: {}'.format(time.time()-t))
	print('Solution found:')
	print(solution)
	print('Number of filled out cells: {}'.format(l))
	print('Original field:\n--------')
	print(sudoku)

	plt.figure()
	plt.plot(bests)
	plt.title('Fitness over iterations')
	plt.xlabel('#Iterations')
	plt.ylabel('Fitness')
	plt.figure()
	plt.title('Distribution of solution lengths')
	plt.hist(lengths)
	plt.xlabel('Length of found solution per ant and iteration')
	plt.ylabel('Number')
	
	##Optimize parameters
	f, ax = plt.subplots(1,3,sharex=True,sharey=True)
	f.suptitle('Convergence for different parameter settings')
	for rho in (.1,.3,.5,.7,.9):
		for i,q in enumerate((.5,1,2)):
			_,_,_,best = run_aco(sudoku, n_ants, q, rho, 50)
			ax[i].plot(best,label='r={}'.format(rho))

	ax[0].set_title('q=0.5')
	ax[0].set_xlabel('#Iterations')
	ax[0].set_ylabel('Fitness')
	ax[1].set_title('q=1.0')
	ax[2].set_title('q=2.0')
	plt.legend()
	
	##Test convergence for different Sudokus
	n = 100
	times = np.zeros((2,len(fields)))
	solved = np.zeros((len(fields),n))
	for fi, f in enumerate(fields):
		sudoku = read_grid(f)
		t = []
		for i in range(n):
			t1 = time.time()
			_,l,_,_ = run_aco(sudoku, 20, .5, .9, 100)
			t2 = time.time()
			t.append(t2-t1)
			solved[fi,i] = l == len(sudoku)**2
			if solved[fi,i]:
				print('Solution found in {}s'.format(t2-t1))
			else:
				print('No Solution found in {}s'.format(t2-t1))	

		times[0,fi] = np.mean(t)
		times[1,fi] = np.std(t)


	pickle.dump(times,open('times.pkl','wb'))
	plt.figure()
	plt.bar(np.arange(times.shape[1]),times[0], yerr=times[1]/np.sqrt(n))
	plt.title('Time per Sudoku')
	plt.xticks(np.arange(times.shape[1]),['10a','10b','11a','11b'])
	plt.xlabel('Sudoku')
	plt.ylabel('Mean time in seconds')

	plt.figure()
	plt.bar(np.arange(solved.shape[0]),np.sum(solved,1)/n)
	plt.title('Fraction solved per Sudoku')
	plt.xticks(np.arange(solved.shape[0]),['10a','10b','11a','11b'])
	plt.xlabel('Sudoku')
	plt.ylabel('Fraction solved')
	plt.show()
	