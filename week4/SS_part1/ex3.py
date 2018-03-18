import numpy as np
import matplotlib.pyplot as plt

def p_not(N):
    """
    :param N: Number of observations
    :return: probability of an observation not being chosen in bootstrapping
    """
    return (1-1/N)**N


N = 30
all_probs = p_not(np.linspace(1,N,N))

plt.plot(range(1,N+1),all_probs, color=[0.8, 0.2, 0.2])
plt.xlabel("Sample size (N)")
plt.ylabel("Probability of observation not being chosen")

