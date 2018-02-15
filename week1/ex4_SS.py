import numpy as np
import matplotlib.pyplot as plt
import copy




def GA(f, n, maxiter):
    """"
    Does a (1+1)-Ga approach for the counting ones problem.
    f = objective function.
    n = bit sequence length
    maxiter = maximum of iterations in search
    """

    # Initial values
    p = 1/n # Mutation probability

    # Step a
    x = np.random.randint(2, size=(n,)) # Create the first random sequence (a)
    fitness_best = f(x) # With fitness

    fitness_array = np.empty((maxiter, 1))  # Storage for plotting best fitness as function of iterations
    fitness_array.fill(np.nan)

    for i in range(1, maxiter+1):

        probs = np.random.uniform(0,1,n) # Create a n-length sequence of probabilities to be used

        x_m = copy.copy(x)

        # Locations of the bit sequence where mutation satisfies the probability
        mut_locs = np.where(probs <= p)[0]

        # Point mutation (invert). The "+ 0" makes boolean integer again
        x_m[mut_locs] = np.logical_not(x_m[mut_locs]) + 0
        
        # Fitness for the mutated x
        fitness_curr = f(x_m)

        if fitness_curr > fitness_best: # If new fitness is better, replace old once
            fitness_best = copy.copy(fitness_curr)
            x = copy.copy(x_m) # Set new x as alpha-x bit sequence

        fitness_array[i-1] = fitness_best  #Store current best fitness in array

        if fitness_best == n:  # If the current best fitness is equal to maximum possible of sum ones
            print("It took", str(i), "iterations to find the maximum, using (1+1)-GA.")
            return x, fitness_best, fitness_array


    print("Maximum iterations reached. Best fitness:",  str(fitness_best))
    return x, fitness_best, fitness_array



# With execution parameters
n = 100 # Sequence length
maxiter = 1500
f = sum # The objective function

# Run run times
total_optimum = 0 # Counting how many times the optimum is reached
runs = 10
best_fitnesses = np.zeros((runs,1)) # Store all best for plotting purposes
fitness_arrays = np.zeros((maxiter,runs)) # For plotting all 10 runs

for i in range(0,runs):
    bit, fit, fitness_array = GA(f, n, maxiter)
    fitness_arrays[:,i] = fitness_array.flatten() # Store fitness array (best fitness per iteration)

    best_fitnesses[i] = fit

    if fit == n: total_optimum += 1 # Count if optimum is reached

print("Number of optimums found:", str(total_optimum) +". Out of", str(runs),"runs.")


# Plotting for question part a
plt.plot(range(maxiter),fitness_arrays);
plt.xlabel("Iteration");
plt.ylabel("Best fitness");
plt.title("Fitness as a function of iteration using (1+1)-GA")
