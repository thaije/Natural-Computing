import numpy as np
from matplotlib import pyplot as plt


def counting_ones_ga(f,n,it):
	'''
	Function solving the counting ones problem
	using a genetic algorithm

	Parameters
	--------
	f : function
		Function to optimize
	n : int
		Length of array
	it : int
		Number of iterations until stop

	Returns
	--------
	array_like
		Array with fitness per iteration	
	'''

	#Create initial random array
	x = np.random.randint(0,2,n)
	p = 1./n
	fit = []

	#For number of iterations
	for i in range(it):
		x_m = np.copy(x)
		#Create indices to flip
		ind = np.random.rand(n) < p
		#Flip bits at indices
		x_m[ind] ^= 1 
		
		#If fitness increases keep new array
		if f(x) < f(x_m):
			x = x_m
		fit.append(f(x))

	return fit

def counting_ones(f, n, it):
	'''
	Function solving the counting ones problem
	using Monte Carlo search

	Parameters
	--------
	f : function
		Function to optimize
	n : int
		Length of array
	it : int
		Number of iterations until stop

	Returns
	--------
	tuple
		contains:
			array_like
				Array with fitness per iteration
			int
				Index of highest fitness	
			
	'''

	#Create it random sequences
	runs = [np.random.randint(0,2,n) for x in range(it)]
	#Calculate fitness per sequence
	fit = np.array(map(f,runs))
	#Find highes fitness
	best = np.argmax(fit)
	return fit, best

#Initialize Paramters
n = 100
it = 1500
f = np.sum

#Plot comparison of convergences of algorithms
fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
fig.suptitle('Counting Ones')

fit,b = counting_ones(f,n,it)

print('Best run using Monte Carlo search was {}'.format(fit[b]))
ax[0].plot([np.max(fit[:x]) for x in range(1,len(fit))])
ax[0].set_title('MC Fitness')
ax[0].set_ylim([40,101])


fit = counting_ones_ga(f,n,it)

print('Best run using Genetic Algorithm was {}'.format(fit[-1]))
ax[1].plot(fit)
ax[1].set_title('GA Fitness')

fig.text(0.5, 0.04, 'Iterations', ha='center')
fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical')

#Run algorithms 100 times and count how many have converged

num = 0
res_mc = []
for i in range(100):
	fit,b = counting_ones(f,n,it)
	res_mc.append(fit[b])
	if fit[b] == n:
		num += 1

print('Number of optimal solutions found using Monte Carlo search was {}'.format(num))

num = 0 
res_ga = []
for i in range(100):
	fit = counting_ones_ga(f,n,it)
	res_ga.append(fit[-1])
	if fit[-1] == n:
		num += 1

print('Number of optimal solutions found using Genetic Algorithm was {}'.format(num))

#Plot histogram of best final results per algorithms
f = plt.figure()
f.suptitle('Best results per algorithm')

bins = np.linspace(50, 100, 50)

plt.hist(res_mc, bins, alpha=0.5, label='MC')
plt.hist(res_ga, bins, alpha=0.5, label='GA')
plt.xlabel('Result')
plt.ylabel('Count')
plt.legend()

plt.show()