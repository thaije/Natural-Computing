import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pdb

data = pd.read_csv('mario.log',delim_whitespace=True,names=['Gen','Species','Genome','Fitness','Maxfitness'],usecols=[1,3,5,7,9])


avg_fitness_generation = []
max_fitness_generation = []
generations = np.unique(data['Gen'])
for gen in generations:
	avg_fitness_generation.append(np.mean(data.loc[data['Gen']==gen]['Fitness']))
	max_fitness_generation.append(np.max(data.loc[data['Gen']==gen]['Fitness']))



plt.figure()
plt.title('Fitness over Generations')
plt.plot(generations,avg_fitness_generation, label='Average')
plt.plot(generations,max_fitness_generation, label='Maximum')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()

plt.figure()
plt.title('Travelled Distance')
plt.hist(data['Fitness'],bins=100)
plt.xlabel('Distance')
plt.ylabel('#Runs')
plt.xlim([-50,250])

plt.figure()
#fitness_species = []
species = np.unique(data['Species'])
for s in species:
	plt.plot(np.array(data.loc[data['Species'] == s]['Gen']),np.cumsum(data.loc[data['Species'] == s]['Fitness']))

#pdb.set_trace()

plt.title('Fitness per Species')
#plt.plot(fitness_species)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()