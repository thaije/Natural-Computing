import numpy as np
from gplearn import functions, fitness, genetic
from matplotlib import pyplot as plt

#train data
x = np.arange(-1,1.1,.1)
#train labels
y = np.array([0, -.1629, -.2624, -.3129, -.3264, -.3125, -.2784, -.2289, -.1664, -.0909, 0, .1111, .2496, .4251, .6496, .9375, 1.3056, 1.7731, 2.3616, 3.0951, 4.])
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#Define exp
def _exp(x):
	y = np.exp(x)
	#protext for infinites
	y[np.isinf(y)] = 10**6
	return y

exp = functions.make_function(_exp, 'exp', 1)

function_set = ['add', 'sub', 'mul','div', 'log', 'sin', 'cos', exp]

#create summed absolute error as metric
_sae = lambda y,t,w: np.sum(np.abs(y-t))
sae = fitness.make_fitness(_sae, False)

n_generations = 50
#Initialize genetic programm regressor
est_gp = genetic.SymbolicRegressor(population_size=1000,
                           generations=1, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0,
                           p_hoist_mutation=0, p_point_mutation=0,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0, random_state=0,
                           metric=sae, function_set=function_set)
est_gp.fit(x, y)

#Generate generations and save best fitness and size
fitness = []
size = []
for i in range(2,n_generations+1):
	est_gp.set_params(generations=i, warm_start=True)
	est_gp.fit(x, y)
	fitness.append(est_gp._program.raw_fitness_)
	size.append(est_gp._program.length_)

#Plot fitness and size over generations
fig, ax = plt.subplots(1,2, sharex=True)
fig.suptitle('Genetic Programming')

ax[0].plot(fitness)
ax[0].set_title('Fitness over generations')
ax[0].set_ylabel('Fitness') 
ax[1].plot(size)
ax[1].set_title('Size over generations')
ax[1].set_ylabel('Size')

fig.text(0.5, 0.04, 'Generation', ha='center')

plt.show()