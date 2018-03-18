import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def major(c, p):
    """One simulation of c agents making a decision with  probability p"""
    return np.sum(np.random.uniform(0,1,c) <= p) >= np.floor(c/2 +1)

def sim(c,p,N_sim):
    """
    Calculates the probability of a majority vote making a correct decision using simulations
    :param c: cases
    :param p: probability
    :param N_sim: number of simulations
    :return: probability of correct decision
    """
    votes = [major(c, p) for i in range(N_sim)]
    p_major = np.mean(votes)
    return p_major

p = 0.6 # Probability of correct decision
c = 21 # Number of medical students
k = np.floor(c/2 +1) # At least majority vote

# Question C
N_sim = 70000
votes = [major(c, p) for i in range(N_sim)]
p_major = np.mean(votes)
print("Probability of the majority making the correct decision is:", str(p_major),"| Based on", str(N_sim),"simulations")


# Question D
N_sim = 10000
P = np.linspace(0.1,0.9,9)
C = range(1,22)


p_matrix = np.zeros((len(P), len(C)))

for p in range(len(P)):
    for c in range(len(C)):
        p_matrix[p][c] = sim(C[c], P[p], N_sim)

# Plotting
ax = sns.heatmap(p_matrix,xticklabels=C, yticklabels=P, cbar_kws={'label': 'Probability majority'},cmap="YlGnBu")
plt.title("Probability of majority vote making correct decision")
plt.xlabel("Number of persons (c)")
plt.ylabel("Probability of correct decision individual (p)")
plt.tight_layout()


# E
p_docs = p_matrix[7][2] # Prob doctors
p_meds = p_matrix[5][20] # Prob medical students

text_doc = "Probability of the doctors making correct majority vote:" + str(p_docs)
text_meds = "Probability of the medical students making correct majority vote:" + str(p_meds)

print(text_doc)
print(text_meds)
if p_docs > p_meds:
    print("The doctors have the highest chance of making the correct decision.")
else:
    print("The medical students have the highest chance of making the correct decision.")



# When meds win
N_sim = 10000
P = np.linspace(0.1,0.9,9)
C = range(1,60)

p_matrix = np.zeros((len(P), len(C)))

for p in range(len(P)):
    for c in range(len(C)):
        p_matrix[p][c] = sim(C[c], P[p], N_sim)

meds_win = [p_matrix[5][i] > p_matrix[7][2] for i in range(len(C))]
print("Medical students need at least", str(C[meds_win.index(True)]),"persons to equal the doctors.")