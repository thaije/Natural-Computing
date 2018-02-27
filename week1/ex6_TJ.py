import matplotlib.pyplot as plt
import numpy as np
import operator, math, random

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

import sys

x = np.linspace(-1., 1., 21)
y = [0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784, -0.2289, -0.1664, -0.0909, 0, 0.1111, 0.2496, 0.4251, 0.6496, 0.9375
, 1.3056, 1.7731, 2.3616, 3.0951, 4.000] # Y-values from exercise 6


plt.plot(x, y, color=[0.2, 0.2, 0.8])
plt.xlabel("x")
plt.ylabel('y')
plt.title('Data exercise 6')
# plt.show()

# Define new functions
def protectedDiv(left, right):
    try:
        return float(left / right)
    except ZeroDivisionError:
        return 1

def protectedLog(val):
    try:
        return math.log(val)
    except ValueError:
        return 1

# define operators
pset = gp.PrimitiveSet("main", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(protectedDiv, 2)

# rename argument as X
pset.renameArguments(ARG0="x")

# define and register individuals
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),
toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])


toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()
