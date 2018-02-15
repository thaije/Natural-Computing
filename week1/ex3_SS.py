import numpy as np

def Monte_Counto(f, n, maxiter):
    """"
    Does a Monte-Carlo search on the Counting Ones problem.
    f = objective function.
    n = bit sequence length
    maxiter = maximum of iterations in search
    """
    # Initialize
    best_seq = None
    best_fitness = 0 # Starting best fitness dummy

    for i in range (1, maxiter+1):
        bit_seq = np.random.randint(2, size=(n,))# Create random "bitstring", an array is easier for computation

        fitness_curr = f(bit_seq) # Calculate fitness of the array: sum of all ones

        if fitness_curr > best_fitness: # If current fitness is the new best
            best_fitness = fitness_curr # Replace old best fitness
            best_seq = bit_seq # Replace old best sequence

        if best_fitness == n: # If the current best fitness is equal to maximum possible of sum ones
            print("It took", str(i), "iterations to find the maximum, using Monte Carlo.")
            return best_seq, best_fitness

    print("Maximum iterations reached. Best fitness:", str(best_fitness))
    return best_seq, best_fitness


# With execution
n = 100
f = sum
maxiter = 1500

# Run
total_optimum = 0 # Counting how many times the optimum is reached
runs = 10

for i in range(1,runs+1):
    bit, fit = Monte_Counto(f, n, maxiter)

    if fit == n: total_optimum += 1 # Count if optimum is reached

print("Number of optimums found:", str(total_optimum) +". Out of", str(runs),"runs.")
