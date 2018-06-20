import matplotlib.pyplot as plt
import pickle
import numpy as np


filename = "models/final_params"
samples = 600
rang = [-200, 1300]
bin_steps = 100



# read in params from loaded model
cum_dists = []
with open (filename, 'rb') as fp:
    [_, _, _, _, cum_dists, _, _] = pickle.load(fp)


# take the last x samples
dist_subset = cum_dists[-samples:]

# create and fill bins
bins = list(range(rang[0], rang[1] + bin_steps, bin_steps))
freqs, bins = np.histogram(dist_subset, bins)
bins = list(bins)
freqs = list(freqs)
freqs.append(0)

print(bins)

freqs = list(freqs)

print(freqs)


pos = np.arange(len(bins))

ax = plt.axes()
ax.set_xticklabels(bins)
ax.set_xticks(pos)

# print(len(bins), len(freq))

plt.bar(pos, freqs, 0.5)


plt.xlabel('Distance')
plt.ylabel('Count')
plt.title(r'Distance per death of last ' + str(samples) + " deaths")
plt.show()
